from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter

from src.data.dto import AssistantMessage, ThreadCreate
from ..util.main import strict_uuid_parser


def get_thread_by_id(thread_id: UUID) -> str:
    """Get thread ID from request"""
    pass


def create_assistant_message(role: str, content: str, status: str = "done") -> AssistantMessage:
    """Create an assistant-ui compatible message"""
    return AssistantMessage(
        id=str(uuid4()),
        role=role,
        content=content,
        createdAt=datetime.now(datetime.UTC),
        status=status
    )


async def stream_assistant_response(
        thread_id: str,
        user_message: str,
        message_id: str
) -> AsyncGenerator[str, None]:
    """Stream responses in assistant-ui format"""
    try:
        # Get thread history
        thread_messages = threads_storage.get(thread_id, [])

        # Convert to LangChain messages
        lc_messages = []
        for msg in thread_messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=str(msg.content)))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=str(msg.content)))

        # Add current user message
        lc_messages.append(HumanMessage(content=user_message))

        # Create initial state
        initial_state = AgentState(
            messages=lc_messages,
            thread_id=thread_id,
            current_message_id=message_id
        )

        # Stream start event
        yield f"data: {json.dumps({'type': 'message-start', 'messageId': message_id})}\n\n"

        collected_content = ""

        # Stream the agent execution
        async for chunk in agent.astream(initial_state.dict()):
            if "messages" in chunk:
                last_message = chunk["messages"][-1]

                if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                    # Stream content delta
                    new_content = last_message.content
                    delta = new_content[len(collected_content):]

                    if delta:
                        collected_content = new_content
                        delta_event = {
                            "type": "text-delta",
                            "messageId": message_id,
                            "textDelta": delta
                        }
                        yield f"data: {json.dumps(delta_event)}\n\n"

                # Handle tool calls
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_event = {
                            "type": "tool-call",
                            "messageId": message_id,
                            "toolCall": {
                                "id": tool_call.get("id"),
                                "name": tool_call.get("name"),
                                "args": tool_call.get("args", {})
                            }
                        }
                        yield f"data: {json.dumps(tool_event)}\n\n"

        # Store the final assistant message
        if collected_content:
            assistant_msg = create_assistant_message("assistant", collected_content)
            if thread_id not in threads_storage:
                threads_storage[thread_id] = []
            threads_storage[thread_id].append(assistant_msg)

        # Stream completion event
        completion_event = {
            "type": "message-complete",
            "messageId": message_id
        }
        yield f"data: {json.dumps(completion_event)}\n\n"

    except Exception as e:
        logger.error(f"Error in stream_assistant_response: {str(e)}")
        error_event = {
            "type": "error",
            "messageId": message_id,
            "error": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"


router = APIRouter(
    prefix="/threads",
    tags=["threads"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.post("/")
async def create_thread(request: ThreadCreate):
    """Create a new thread (assistant-ui compatible)"""
    thread_id = str(uuid4())

    # Convert messages to AssistantMessage format
    thread_messages = []
    for msg in request.messages:
        assistant_msg = create_assistant_message(
            role=msg.role,
            content=msg.content
        )
        thread_messages.append(assistant_msg)

    threads_storage[thread_id] = thread_messages

    return {
        "id": thread_id,
        "createdAt": datetime.utcnow().isoformat(),
        "messages": [msg.dict() for msg in thread_messages]
    }


@router.get("/{thread_id}")
async def get_thread(thread_id: str):
    """
    Get thread by ID
    """
    thread = get_thread_by_id(strict_uuid_parser(thread_id))

    messages = threads_storage[thread_id]
    return {
        "id": thread_id,
        "messages": [msg.dict() for msg in messages]
    }


@router.post("/{thread_id}/messages")
async def append_message(thread_id: str, request: AppendMessageRequest):
    """Add message and stream response (assistant-ui compatible)"""
    if thread_id not in threads_storage:
        # Create thread if it doesn't exist
        threads_storage[thread_id] = []

    # Add user message to thread
    user_msg = create_assistant_message(
        role=request.role,
        content=request.content
    )
    threads_storage[thread_id].append(user_msg)

    # Generate assistant message ID
    assistant_message_id = str(uuid.uuid4())

    # Create in-progress assistant message
    assistant_msg = create_assistant_message(
        role="assistant",
        content="",
        status="in_progress"
    )
    assistant_msg.id = assistant_message_id
    threads_storage[thread_id].append(assistant_msg)

    # Return streaming response
    return StreamingResponse(
        stream_assistant_response(
            thread_id=thread_id,
            user_message=str(request.content),
            message_id=assistant_message_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/{thread_id}/messages")
async def get_messages(thread_id: str):
    """Get all messages in a thread (assistant-ui compatible)"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = threads_storage[thread_id]
    return {"messages": [msg.dict() for msg in messages]}


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread (assistant-ui compatible)"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")

    del threads_storage[thread_id]
    return {"success": True}


# Assistant-UI configuration endpoint
@app.get("/config")
async def get_config():
    """Return assistant configuration for assistant-ui"""
    return {
        "runtime": {
            "isRunning": True,
            "capabilities": {
                "switchToBranch": False,
                "edit": False,
                "reload": True,
                "cancel": False,
                "unstable_copy": False
            }
        },
        "assistantId": "langgraph-agent",
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            } for tool in tools
        ]
    }
