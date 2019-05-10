import pdb

if __name__ == '__main__':
    import multiagent.scenarios as scenarios
    scenario = scenarios.load("multiagent-particle-envs/multiagent/scenarios/simple_tag.py").Scenario()
    world = scenario.make_world()
    from multiagent.environment import MultiAgentEnv
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    n_agents = env.n; dim_act = world.dim_p * 2 + 1
    obs = env.reset(); n_states = len(obs[0] )

    n_episode = 20000; max_steps = 1200
    from maddpg import *
    maddpg = MADDPG(n_agents, n_states, dim_act )

    for i_episode in range(n_episode):
        obs = env.reset(); obs = np.stack(obs)
        max_steps = 1200; total_reward = 0
        adversaries_reward = 0; goodagent_reward = 0
        for t in range(max_steps ):
            actions = maddpg.produce_action(obs )
            obs_, reward, done, _ = env.step(actions.detach() )
            next_obs = None

            if t < max_steps - 1:
                next_obs = obs_

            for r in reward:
                total_reward += r

            adversaries_reward += (reward[0] + reward[1] + reward[2] )
            goodagent_reward += reward[3]
            maddpg.memory.push(obs, actions, next_obs, reward)
            obs = next_obs;
            maddpg.train(i_episode); env.render()

        print('Episode: %u' % (i_episode + 1) )
        print('总体累积奖赏值 = %f' % (total_reward) )
        print('adversary得到的累积奖赏值 = %f' % adversaries_reward )
        print('good agent得到的累积奖赏值 = %f\n' % goodagent_reward )
        maddpg.episode_done += 1