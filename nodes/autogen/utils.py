from autogen import ConversableAgent

def clone_conversable_agent(source_agent: ConversableAgent) -> ConversableAgent:
    # Create a new agent by calling the __init__ method with the source agent's parameters
    cloned_agent = ConversableAgent(
        name=source_agent._name,
        system_message=source_agent._oai_system_message[0]["content"],
        max_consecutive_auto_reply=source_agent._max_consecutive_auto_reply,
        human_input_mode=source_agent.human_input_mode,
        function_map=source_agent._function_map,
        code_execution_config=source_agent._code_execution_config,
        llm_config=source_agent.llm_config,
        default_auto_reply=source_agent._default_auto_reply,
        description=source_agent._description,
        is_termination_msg=lambda msg: "terminate" in msg["content"].lower() if msg["content"] is not None else False,
    )

    # Manually copy other relevant attributes
    cloned_agent._oai_messages = source_agent._oai_messages.copy()
    cloned_agent._consecutive_auto_reply_counter = source_agent._consecutive_auto_reply_counter.copy()
    cloned_agent._max_consecutive_auto_reply_dict = source_agent._max_consecutive_auto_reply_dict.copy()
    cloned_agent.reply_at_receive = source_agent.reply_at_receive.copy()
    cloned_agent._reply_func_list = source_agent._reply_func_list.copy()
    cloned_agent._human_input = source_agent._human_input.copy()
    cloned_agent.client_cache = source_agent.client_cache
    cloned_agent.client = source_agent.client
    cloned_agent.hook_lists = source_agent.hook_lists.copy()

    for i in cloned_agent._oai_messages:
        i._oai_messages[cloned_agent] = i._oai_messages[source_agent].copy()

    return cloned_agent
