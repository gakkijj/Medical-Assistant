"""
SwarmCoordinator：Swarm 入口和智能路由

注意：这不是编排器！
- 只负责路由决策：简单问题 → 单 Agent，复杂问题 → Swarm
- 不控制 Agent 执行
- 不编排任务顺序

类比：交通信号灯，决定车辆走哪条路，但不控制车辆如何行驶
"""
import asyncio
import time
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from core import AdaptiveRouter, LLMClient, RouteDecision
from core.request_metrics import record_route_decision
from .shared_context import SharedContext
from .lead_agent import LeadAgent
from .events import Event, EventType
from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from memory import SessionSummaryManager, SessionSummary, ShortTermMemory, LongTermMemory


class SwarmCoordinator:
    """
    Swarm 协调器

    职责：
    1. 智能路由（简单 → 单 Agent，复杂 → Swarm）
    2. 初始化 SharedContext
    3. 启动和监控 Swarm
    4. 生成 SessionSummary

    不做：
    - 不编排 Worker 执行顺序
    - 不直接调用 Worker
    - 不控制任务分配
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True,
        router: Optional[AdaptiveRouter] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm
        self.router = router or AdaptiveRouter()

        # 初始化 Agent
        self.lead_agent = LeadAgent(llm_client=self.llm_client)
        self.consultation_agent = ConsultationAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.research_agent = ResearchAgent()

        # Worker 池
        self.worker_pool: List[Any] = [
            self.consultation_agent,
            self.diagnostic_agent,
            self.research_agent
        ]

        # 记忆管理器
        self.session_manager = SessionSummaryManager()
        self.short_term_memory = ShortTermMemory(storage_type="memory")  # 或 "redis"
        self.long_term_memory = LongTermMemory()

        # 将短期记忆注入到所有 Worker Agent 的 Loop
        # 注意：LeadAgent 不继承 BaseAgent，没有 loop 属性，不需要注入
        for worker in self.worker_pool:
            if hasattr(worker, 'loop'):
                worker.loop.short_term_memory = self.short_term_memory

        logger.info(f"SwarmCoordinator initialized with {len(self.worker_pool)} workers")
        logger.info(f"Memory system: short_term={self.short_term_memory.storage_type}, long_term={'enabled' if self.long_term_memory.enabled else 'disabled'}")

    def _get_agent_by_id(self, agent_id: str):
        """根据 agent_id 返回对应的 Agent 实例"""
        mapping = {
            "consultation_agent": self.consultation_agent,
            "diagnostic_agent": self.diagnostic_agent,
            "research_agent": self.research_agent
        }
        return mapping.get(agent_id)

    @staticmethod
    def _build_router_assessment(
        question: str,
        decision: RouteDecision,
    ) -> Dict[str, Any]:
        """Build deterministic fallback tasks when LLM planning is unavailable."""
        descriptions = {
            "consultation_agent": f"为用户问题提供清晰、可执行的健康建议：{question}",
            "diagnostic_agent": f"评估症状风险、紧急程度并给出鉴别思路：{question}",
            "research_agent": f"检索指南和证据，标明来源与适用范围：{question}",
        }
        return {
            "subtasks": [
                {
                    "description": descriptions[agent_id],
                    "assigned_agent": agent_id,
                }
                for agent_id in decision.recommended_agents
            ],
            "reason": "adaptive_router_fallback",
        }

    def _ensure_swarm_assessment(
        self,
        question: str,
        assessment: Dict[str, Any],
        decision: RouteDecision,
    ) -> Dict[str, Any]:
        valid_agents = {
            task.get("assigned_agent")
            for task in assessment.get("subtasks", [])
            if self._get_agent_by_id(task.get("assigned_agent")) is not None
        }
        if len(valid_agents) >= 2:
            return assessment
        logger.warning("LeadAgent returned fewer than two valid workers; using router fallback")
        return self._build_router_assessment(question, decision)

    def _resolve_llm_fallback(
        self,
        decision: RouteDecision,
        assessment: Dict[str, Any],
    ) -> RouteDecision:
        """Use the first valid LeadAgent assignment to resolve a low-confidence tie."""
        selected_agent = next((
            task.get("assigned_agent")
            for task in assessment.get("subtasks", [])
            if self._get_agent_by_id(task.get("assigned_agent")) is not None
        ), None)
        if not selected_agent:
            return decision
        intent_by_agent = {
            "consultation_agent": "consultation",
            "diagnostic_agent": "diagnosis",
            "research_agent": "research",
        }
        return replace(
            decision,
            intent=intent_by_agent[selected_agent],
            primary_agent=selected_agent,
            recommended_agents=(
                decision.recommended_agents
                if decision.mode == "swarm"
                else (selected_agent,)
            ),
            confidence=max(decision.confidence, 0.75),
            reason_codes=tuple(decision.reason_codes) + ("llm_fallback_applied",),
            fallback_required=False,
            router_stage="llm_fallback",
            lead_planning_required=True,
        )

    @staticmethod
    def _collect_citations(shared_context: SharedContext) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for contribution in shared_context.get_contributions():
            result = contribution.result if isinstance(contribution.result, dict) else {}
            for citation in result.get("citations", []):
                if citation not in citations:
                    citations.append(citation)
        return citations

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户问题

        Args:
            question: 用户问题
            context: 额外上下文（年龄、既往史等）
            session_id: 会话ID（如果不提供，将自动生成）

        Returns:
            处理结果
        """
        start_time = datetime.now()
        if session_id is None:
            session_id = f"{start_time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

        logger.info(f"Processing question session={session_id} length={len(question)}")

        # ===== 统一的记忆检索（所有模式都使用）=====
        # 1. 检索短期记忆（当前会话历史）
        recent_history = self.short_term_memory.get_recent_messages(
            session_id=session_id,
            limit=10  # 最近5轮对话（10条消息）
        )

        # 2. 检索长期记忆（相似历史会话）
        similar_memories = self.long_term_memory.search_similar_sessions(
            query=question,
            limit=3,
            user_id=session_id
        )

        # 3. 构建增强上下文
        enhanced_context = dict(context or {})
        force_mode = enhanced_context.pop("_routing_mode", None)

        # 添加短期记忆
        if recent_history:
            enhanced_context["recent_history"] = [
                {"role": msg.get("role", ""), "content": msg.get("content", "")}
                for msg in recent_history
            ]
            logger.info(f"Loaded {len(recent_history)} recent messages from short-term memory")

        # 添加长期记忆
        if similar_memories:
            enhanced_context["historical_cases"] = [
                {
                    "summary": mem["content"],
                    "score": mem["score"]
                }
                for mem in similar_memories
            ]
            logger.info(f"Found {len(similar_memories)} similar historical cases from long-term memory")

        # Step 1: zero-LLM adaptive routing. Simple requests skip LeadAgent.
        route_start = time.perf_counter()
        decision = self.router.decide(
            question,
            has_history=bool(recent_history),
            enable_swarm=self.enable_swarm,
            force_mode=force_mode,
        )
        if decision.mode == "swarm":
            assessment = await self.lead_agent.assess_and_decompose(question, enhanced_context)
            assessment = self._ensure_swarm_assessment(question, assessment, decision)
            if decision.fallback_required:
                decision = self._resolve_llm_fallback(decision, assessment)
        elif decision.fallback_required:
            fallback_assessment = await self.lead_agent.assess_and_decompose(
                question,
                enhanced_context,
            )
            decision = self._resolve_llm_fallback(decision, fallback_assessment)
            assessment = self._build_router_assessment(question, decision)
        else:
            assessment = self._build_router_assessment(question, decision)
        route_data = decision.to_dict()
        record_route_decision(route_data, time.perf_counter() - route_start)
        logger.info(
            f"Adaptive route: mode={decision.mode}, primary={decision.primary_agent}, "
            f"score={decision.complexity_score}, reasons={decision.reason_codes}"
        )
        subtasks = assessment.get("subtasks", [])

        # Step 2: 根据任务数量路由
        final_answer = None
        mode = None

        if decision.mode == "single":
            # 单任务 → 直接调用对应 Agent
            task = subtasks[0]
            agent_id = task.get("assigned_agent")
            agent = self._get_agent_by_id(agent_id)

            if agent is None:
                # 如果找不到 Agent，降级到 ConsultationAgent
                logger.warning(f"Unknown agent_id: {agent_id}, fallback to ConsultationAgent")
                agent = self.consultation_agent

            logger.info(f"Route: Single Agent ({agent_id})")
            mode = "single_agent"
            result = await agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')

            result.update({
                'swarm_enabled': False,
                'session_id': session_id,
                'route_reason': f'自适应路由到 {agent_id}',
                'route': route_data,
            })

            # 确保单Agent模式下也有 disclaimer 字段
            if 'disclaimer' not in result:
                result['disclaimer'] = "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"

            # 确保单Agent模式下也有 suggestions 字段
            if 'suggestions' not in result:
                result['suggestions'] = []

        elif decision.mode == "swarm" and self.enable_swarm:
            # 多任务 → 启动 Swarm
            logger.info(f"Route: Swarm (Multi-Agent Collaboration) - {len(subtasks)} tasks")
            mode = "swarm"
            result = await self._process_with_swarm(
                question=question,
                context=enhanced_context,
                assessment=assessment,
                session_id=session_id,
                start_time=start_time,
                route_decision=decision,
            )
            final_answer = result.get('answer', '')

            # Swarm 模式已经在 _process_with_swarm 中保存了长期记忆，直接返回
            return result

        else:
            # 0个任务或Swarm关闭 → 降级到 ConsultationAgent
            if len(subtasks) == 0:
                logger.warning("No subtasks generated, fallback to ConsultationAgent")
                mode = "fallback"
            else:
                logger.info("Swarm disabled, fallback to ConsultationAgent")
                mode = "disabled_swarm"

            result = await self.consultation_agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')
            result.update({
                'swarm_enabled': False,
                'session_id': session_id,
                'route': route_data,
            })

        # ===== 统一的记忆保存（非 Swarm 模式）=====
        end_time = datetime.now()

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存

        # 保存到长期记忆
        try:
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                metadata={
                    "mode": mode,
                    "subtasks_count": len(subtasks),
                    "total_time": (end_time - start_time).total_seconds(),
                },
                user_id=session_id
            )
            logger.info(f"Saved to long-term memory (session={session_id}, mode={mode})")
        except Exception as e:
            logger.error(f"Failed to save to long-term memory: {e}")

        return result

    async def _process_with_swarm(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        assessment: Dict[str, Any],
        session_id: str,
        start_time: datetime,
        route_decision: RouteDecision,
    ) -> Dict[str, Any]:
        """
        使用 Swarm 处理复杂问题

        这是群体智能的核心流程

        注意：context 已经包含了长短期记忆（在 process() 中注入）
        """
        # context 已经包含 recent_history 和 historical_cases
        # 无需重复检索

        # 创建 SharedContext
        shared_context = SharedContext(session_id=session_id)

        # 发布 Swarm 启动事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="swarm_coordinator",
            data={
                "question": question,
                "num_subtasks": len(assessment.get("subtasks", []))
            }
        ))

        # Step 1: LeadAgent 分解任务
        subtasks = self.lead_agent.create_subtasks(assessment, shared_context)
        logger.info(f"Created {len(subtasks)} subtasks")

        # Step 2: Worker 执行分配的任务（并行）
        tasks = []
        for worker in self.worker_pool:
            task = asyncio.create_task(
                self._worker_execute_assigned_tasks(worker, shared_context)
            )
            tasks.append(task)

        # 等待所有 Worker 完成（或超时）
        timeout_occurred = False
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=90.0  # 增加超时时间到 90 秒，应对复杂案例
            )
        except asyncio.TimeoutError:
            timeout_occurred = True
            logger.warning("Swarm execution timeout (90s)")
            # 记录哪些 Agent 已完成，哪些未完成
            completed_agents = list(shared_context.agent_contributions.keys())
            claimed_tasks = [
                (subtask.assigned_to, subtask.type)
                for subtask in shared_context.task_decomposition.values()
                if subtask.status.value == "claimed"
            ]
            logger.info(f"Completed agents: {completed_agents}")
            logger.info(f"Timed out tasks: {claimed_tasks}")

        # Step 3: LeadAgent 汇总结果
        # 即使超时，也尝试汇总已完成的部分结果
        final_answer = await self.lead_agent.synthesize_results(
            question=question,
            shared_context=shared_context,
            timeout_occurred=timeout_occurred
        )

        end_time = datetime.now()

        # Step 4: 生成 SessionSummary
        try:
            summary = SessionSummary.from_shared_context(
                session_id=session_id,
                question=question,
                shared_context=shared_context,
                final_answer=final_answer,
                start_time=start_time,
                end_time=end_time
            )
            self.session_manager.save_summary(summary)
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存
        # Agent Loop 保存了完整的对话历史（user + assistant + tool messages）

        # 保存到 Mem0 长期记忆
        try:
            # 保存会话总结
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                metadata={
                    "mode": "swarm",
                    "agents_count": len(shared_context.agent_contributions),
                    "total_time": (end_time - start_time).total_seconds(),
                    "timeout_occurred": timeout_occurred
                },
                user_id=session_id
            )

            logger.info(f"Saved to Mem0 long-term memory (session={session_id})")

        except Exception as e:
            logger.error(f"Failed to save to Mem0: {e}")

        # 发布 Swarm 完成事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="swarm_coordinator",
            data={
                "duration": (end_time - start_time).total_seconds(),
                "agents_count": len(shared_context.agent_contributions)
            }
        ))

        # 返回结果
        completed_agents = list(shared_context.agent_contributions.keys())
        result = {
            'answer': final_answer,
            'swarm_enabled': True,
            'session_id': session_id,
            'agents_involved': completed_agents,
            'subtasks_completed': len(shared_context.get_all_completed_subtasks()),
            'total_time': (end_time - start_time).total_seconds(),
            'swarm_metadata': shared_context.get_summary(),
            'timeout_occurred': timeout_occurred,
            'route': route_decision.to_dict(),
            'citations': self._collect_citations(shared_context),
        }

        # 提取建议和免责声明（简化实现）
        result['suggestions'] = self._extract_suggestions(final_answer)

        # 根据是否超时调整免责声明
        if timeout_occurred and not completed_agents:
            result['disclaimer'] = "由于系统超时，未能提供完整分析。建议简化问题重试，或在紧急情况下立即就医。"
        elif timeout_occurred:
            result['disclaimer'] = f"以上分析基于 {len(completed_agents)} 个 Agent 的部分协作结果（部分分析模块超时未完成），仅供参考，不能替代医生诊断。"
        else:
            result['disclaimer'] = "以上分析基于多个专业 Agent 的协作，仅供参考，不能替代医生诊断。"

        return result

    async def _worker_execute_assigned_tasks(
        self,
        worker: Any,
        shared_context: SharedContext
    ):
        """
        Worker 执行分配给它的任务

        简化后的流程：
        - 查找分配给自己的任务
        - 执行任务
        - 记录结果
        """
        try:
            # 获取分配给该 Agent 的任务
            assigned_tasks = shared_context.get_subtasks_for_agent(worker.agent_id)

            if not assigned_tasks:
                logger.debug(f"{worker.agent_id}: No assigned tasks")
                return

            # 并行执行所有分配的任务
            tasks = []
            for subtask in assigned_tasks:
                logger.info(f"{worker.agent_id}: Starting {subtask.type}")
                shared_context.start_subtask(subtask.id)

                task = asyncio.create_task(
                    self._execute_single_subtask(worker, subtask, shared_context)
                )
                tasks.append(task)

            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"{worker.agent_id}: Error processing subtask: {e}")

    async def _execute_single_subtask(self, worker, subtask, shared_context):
        """执行单个子任务"""
        try:
            result = await worker.process_subtask(subtask, shared_context=shared_context)
            shared_context.complete_subtask(subtask.id, worker.agent_id, result)
            logger.info(f"{worker.agent_id}: Completed {subtask.type}")
        except Exception as e:
            logger.error(f"{worker.agent_id}: Error in {subtask.type}: {e}")

    def _extract_suggestions(self, final_answer: str) -> List[str]:
        """从最终答案中提取建议（简化实现）"""
        suggestions = []

        # 简单的文本匹配
        if "【核心建议】" in final_answer:
            # 提取核心建议部分
            start_idx = final_answer.find("【核心建议】")
            end_idx = final_answer.find("【", start_idx + 1)
            if end_idx == -1:
                end_idx = len(final_answer)

            suggestions_text = final_answer[start_idx:end_idx]

            # 提取编号列表
            import re
            matches = re.findall(r'\d+\.\s*([^\n]+)', suggestions_text)
            suggestions = matches[:5]  # 最多5条

        return suggestions or ["请遵循医嘱，注意休息和营养"]

async def process_with_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：使用 Swarm 处理问题

    Args:
        question: 用户问题
        context: 额外上下文
        enable_swarm: 是否启用 Swarm（False 则总是用单 Agent）
        session_id: 会话ID（如果提供，将使用该ID而不是生成新的）

    Returns:
        处理结果
    """
    coordinator = SwarmCoordinator(enable_swarm=enable_swarm)
    return await coordinator.process(question, context, session_id=session_id)
