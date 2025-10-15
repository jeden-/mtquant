"""
Agent API Routes - FastAPI Endpoints.

Provides REST API endpoints for agent management.

Author: MTQuant Development Team
Date: October 15, 2025
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional

from api.models.agent_schemas import (
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentStateTransitionRequest,
    AgentResponse,
    AgentListResponse,
    AgentHealthResponse,
    AgentActionResponse,
    AgentPerformanceResponse,
)
from mtquant.agents.agent_manager import (
    AgentLifecycleManager,
    AgentScheduler,
    AgentRegistry,
    AgentState,
    AgentInfo,
)
from mtquant.utils.exceptions import AgentError, StateTransitionError
from mtquant.utils.logger import get_logger


logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/agents", tags=["agents"])

# Global instances (should be injected via dependency injection in production)
lifecycle_manager: Optional[AgentLifecycleManager] = None
scheduler: Optional[AgentScheduler] = None
registry: Optional[AgentRegistry] = None


def get_lifecycle_manager() -> AgentLifecycleManager:
    """Dependency to get lifecycle manager."""
    if lifecycle_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent management system not initialized"
        )
    return lifecycle_manager


def get_scheduler() -> AgentScheduler:
    """Dependency to get scheduler."""
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not initialized"
        )
    return scheduler


def get_registry() -> AgentRegistry:
    """Dependency to get registry."""
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry not initialized"
        )
    return registry


def _agent_info_to_response(agent_info: AgentInfo) -> AgentResponse:
    """Convert AgentInfo to AgentResponse."""
    return AgentResponse(
        agent_id=agent_info.agent_id,
        agent_type=agent_info.agent_type,
        symbol=agent_info.symbol,
        state=agent_info.state.value,
        created_at=agent_info.created_at,
        last_state_change=agent_info.last_state_change,
        metrics={
            'total_trades': agent_info.metrics.total_trades,
            'winning_trades': agent_info.metrics.winning_trades,
            'losing_trades': agent_info.metrics.losing_trades,
            'total_pnl': agent_info.metrics.total_pnl,
            'sharpe_ratio': agent_info.metrics.sharpe_ratio,
            'max_drawdown': agent_info.metrics.max_drawdown,
            'win_rate': agent_info.metrics.win_rate,
            'last_updated': agent_info.metrics.last_updated
        },
        config=agent_info.config,
        error_message=agent_info.error_message
    )


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Create a new agent.
    
    Args:
        request: Agent creation request
        manager: Lifecycle manager (injected)
        
    Returns:
        Created agent information
        
    Raises:
        HTTPException: If agent already exists or creation fails
    """
    try:
        agent_info = manager.register_agent(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            symbol=request.symbol,
            config=request.config.dict() if request.config else {}
        )
        
        logger.info(f"Created agent {request.agent_id}")
        return _agent_info_to_response(agent_info)
    
    except AgentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    agent_type: Optional[str] = None,
    state: Optional[str] = None,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    List all agents with optional filters.
    
    Args:
        agent_type: Filter by agent type (optional)
        state: Filter by state (optional)
        manager: Lifecycle manager (injected)
        
    Returns:
        List of agents
    """
    try:
        if state:
            try:
                agent_state = AgentState(state)
                agents = manager.get_agents_by_state(agent_state)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid state: {state}"
                )
        elif agent_type:
            agents = manager.get_agents_by_type(agent_type)
        else:
            agents = list(manager.get_all_agents().values())
        
        return AgentListResponse(
            agents=[_agent_info_to_response(agent) for agent in agents],
            total=len(agents)
        )
    
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Get agent information.
    
    Args:
        agent_id: Agent identifier
        manager: Lifecycle manager (injected)
        
    Returns:
        Agent information
        
    Raises:
        HTTPException: If agent not found
    """
    try:
        agent_info = manager.get_agent_info(agent_id)
        return _agent_info_to_response(agent_info)
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent: {str(e)}"
        )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Update agent configuration.
    
    Args:
        agent_id: Agent identifier
        request: Update request
        manager: Lifecycle manager (injected)
        
    Returns:
        Updated agent information
        
    Raises:
        HTTPException: If agent not found or update fails
    """
    try:
        agent_info = manager.get_agent_info(agent_id)
        
        if request.config:
            agent_info.config.update(request.config.dict(exclude_unset=True))
        
        logger.info(f"Updated agent {agent_id}")
        return _agent_info_to_response(agent_info)
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to update agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent: {str(e)}"
        )


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Delete an agent.
    
    Args:
        agent_id: Agent identifier
        manager: Lifecycle manager (injected)
        
    Raises:
        HTTPException: If agent not found or deletion fails
    """
    try:
        manager.unregister_agent(agent_id)
        logger.info(f"Deleted agent {agent_id}")
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to delete agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )


@router.post("/{agent_id}/transition", response_model=AgentActionResponse)
async def transition_agent_state(
    agent_id: str,
    request: AgentStateTransitionRequest,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Transition agent to a new state.
    
    Args:
        agent_id: Agent identifier
        request: State transition request
        manager: Lifecycle manager (injected)
        
    Returns:
        Action response
        
    Raises:
        HTTPException: If agent not found or transition invalid
    """
    try:
        new_state = AgentState(request.new_state)
        
        manager.transition_state(
            agent_id=agent_id,
            new_state=new_state,
            error_message=request.error_message
        )
        
        return AgentActionResponse(
            success=True,
            message=f"Agent transitioned to {request.new_state}",
            agent_id=agent_id,
            new_state=request.new_state
        )
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except StateTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state: {request.new_state}"
        )
    except Exception as e:
        logger.error(f"Failed to transition agent state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transition state: {str(e)}"
        )


@router.post("/{agent_id}/start", response_model=AgentActionResponse)
async def start_agent(
    agent_id: str,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Start an agent (transition to TRAINING or LIVE).
    
    Args:
        agent_id: Agent identifier
        manager: Lifecycle manager (injected)
        
    Returns:
        Action response
    """
    try:
        agent_info = manager.get_agent_info(agent_id)
        
        # Determine target state based on current state
        if agent_info.state == AgentState.INITIALIZED:
            target_state = AgentState.TRAINING
        elif agent_info.state == AgentState.PAUSED:
            # Resume to previous state (simplified - assumes LIVE)
            target_state = AgentState.LIVE
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot start agent in state {agent_info.state.value}"
            )
        
        manager.transition_state(agent_id, target_state)
        
        return AgentActionResponse(
            success=True,
            message=f"Agent started ({target_state.value})",
            agent_id=agent_id,
            new_state=target_state.value
        )
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except StateTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/{agent_id}/pause", response_model=AgentActionResponse)
async def pause_agent(
    agent_id: str,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Pause an agent.
    
    Args:
        agent_id: Agent identifier
        manager: Lifecycle manager (injected)
        
    Returns:
        Action response
    """
    try:
        manager.transition_state(agent_id, AgentState.PAUSED)
        
        return AgentActionResponse(
            success=True,
            message="Agent paused",
            agent_id=agent_id,
            new_state="paused"
        )
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except StateTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/{agent_id}/stop", response_model=AgentActionResponse)
async def stop_agent(
    agent_id: str,
    manager: AgentLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Stop an agent.
    
    Args:
        agent_id: Agent identifier
        manager: Lifecycle manager (injected)
        
    Returns:
        Action response
    """
    try:
        manager.transition_state(agent_id, AgentState.STOPPED)
        
        return AgentActionResponse(
            success=True,
            message="Agent stopped",
            agent_id=agent_id,
            new_state="stopped"
        )
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except StateTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{agent_id}/health", response_model=AgentHealthResponse)
async def get_agent_health(
    agent_id: str,
    reg: AgentRegistry = Depends(get_registry)
):
    """
    Get agent health status.
    
    Args:
        agent_id: Agent identifier
        reg: Registry (injected)
        
    Returns:
        Health status
        
    Raises:
        HTTPException: If agent not found
    """
    try:
        health = reg.get_health_status(agent_id)
        
        return AgentHealthResponse(
            agent_id=health['agent_id'],
            state=health['state'],
            is_healthy=health['is_healthy'],
            last_heartbeat=health['last_heartbeat'],
            uptime_seconds=health['uptime_seconds'],
            error_message=health['error_message']
        )
    
    except AgentError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to get agent health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}"
        )


# Initialize function (should be called on app startup)
def initialize_agent_routes(
    lifecycle_mgr: AgentLifecycleManager,
    sched: AgentScheduler,
    reg: AgentRegistry
):
    """
    Initialize agent routes with manager instances.
    
    Args:
        lifecycle_mgr: Lifecycle manager instance
        sched: Scheduler instance
        reg: Registry instance
    """
    global lifecycle_manager, scheduler, registry
    lifecycle_manager = lifecycle_mgr
    scheduler = sched
    registry = reg
    logger.info("Agent routes initialized")


