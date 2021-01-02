def run_test(envs, policy):
    debug_memories = [[] for _ in envs]
    state = np.array([e.reset()[0] for e in envs])
    total_reward = 0

    for i in range(0, 100):
        action, policy_info = policy.get_actions(state)

        statewrappers = [e.step(a) for a, e in zip(action, envs)]
        
        terminals = np.array([(s.step_type == garage.StepType.TERMINAL or s.step_type == garage.StepType.TIMEOUT) for s in statewrappers])
        rewards = np.array([s.reward for s in statewrappers])
        total_reward += np.sum(rewards)
        
        next_state = np.array([s.observation for s in statewrappers])
        infos = [s.env_info for s in statewrappers]

        for i in range(len(envs)):
            debug_memories[i].append((
                state[i],
                action[i],
                next_state[i],
                rewards[i],
                terminals[i],
                infos[i]
            ))
        
        if np.any(terminals):
            next_state[terminals] = [e.reset()[0] for e,t in zip(envs, terminals) if t]
        
        state = next_state
    
    return debug_memories, total_reward
