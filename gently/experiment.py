

def do_scenario(agent, teacher, contexts):
    for context in contexts:
        action = agent.act(context)

        feedback = teacher.get_feedback(action, context)

        agent.update(feedback, action)
