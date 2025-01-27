"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

import hac.hac.agent as Agent
from hac.utils import print_summary
import wandb
import numpy as np
import torch
import logging
import time

num_test_episodes = 100

def run_HAC(args, env, agent):

    enable_wandb = not args.test

    if enable_wandb:
        wandb.init(project=args.env, settings=wandb.Settings(_disable_stats=True), 
                    entity='ajain23',
                    group=args.group if args.group is not None else '_'.join(["hac", str(args.env), str(args.n_layers)]),
                    name=args.name if args.name is not None else 's' + str(args.seed))


    NUM_BATCH = args.num_batch_train
    TEST_FREQ = args.test_freq

    total_episodes = 0

    start_time = time.time()

    # Print task summary
    print_summary(args,env)
   
    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not args.test and not args.train_only:
        mix_train_test = True
     
    for batch in range(NUM_BATCH):

        num_episodes = agent.other_params["num_exploration_episodes"]

        if agent.total_env_steps > args.timesteps:
            return

        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            logging.info("\n--- TESTING ---")
            agent.args.test = True
            num_episodes = num_test_episodes            

            # Reset successful episode counter
            test_episode_lengths = []
            test_subgoal_achieved_ratio = np.zeros((num_test_episodes, args.n_layers - 1))

            successful_test_episodes = 0

        for episode in range(num_episodes):
            
            logging.info("\nBatch %d, Episode %d" % (batch, episode))

            agent.subgoal_achieved_info = [[0, 0] for i in range(agent.args.n_layers - 1)]
            
            # Train for an episode
            success = agent.train(env, episode, total_episodes)

            if mix_train_test and batch % TEST_FREQ == 0:
                test_episode_lengths.append(agent.steps_taken)

                for i in range(args.n_layers - 1):
                    # Get number of goal proposed/achieved
                    subgoal_info = agent.subgoal_achieved_info[i]
                    subgoal_achieved_ratio = subgoal_info[0] / subgoal_info[1]
                    test_subgoal_achieved_ratio[episode][i] = subgoal_achieved_ratio

            if success:
                logging.info("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                
                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_test_episodes += 1            

            if agent.args.train_only or (mix_train_test and batch % TEST_FREQ != 0):
                total_episodes += 1

        # Save agent
        agent.save_model()
           
        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:

            # Compute performance
            success_rate = successful_test_episodes / num_test_episodes
            average_episode_length = np.mean(test_episode_lengths)

            test_subgoal_achieved_average = test_subgoal_achieved_ratio.mean(axis=0)

            print("Step: %d Test Succ. Rate %.2f%%, Avg. Len.: %.2f" % (agent.total_env_steps, success_rate * 100, average_episode_length))

            for i in range(len(test_subgoal_achieved_average)):
                print(f"Goal Achieved Ratio (Level {i})", test_subgoal_achieved_average[i])

            print("--------------")

            hours = (time.time() - start_time) / 3600

            # Log performance
            if enable_wandb:
                wandb.log({
                    'Success Rate': success_rate, 
                    'Episode Length': average_episode_length,
                    'Hours': hours},
                    step=agent.total_env_steps)

                for i in range(len(test_subgoal_achieved_average)):
                    wandb.log({f"Goal Achieved Ratio (Level {i})": test_subgoal_achieved_average[i]}, step=agent.total_env_steps)

                for i in range(args.n_layers):
                    wandb.log({f"Buffer Size (Level {i})": agent.replay_buffer_sizes[i]}, step=agent.total_env_steps)
            
            agent.args.test = False
            
def test_HAC(args, env, agent):

    assert(args.test)

    test_episodes = 100

    # Reset successful episode counter
    successful_test_episodes = 0
        
    test_episode_lengths = []

    for episode in range(test_episodes):

        logging.info("\nEpisode %d" % (episode))

        agent.subgoal_achieved_info = [[0, 0] for i in range(agent.args.n_layers - 1)]
        
        # Run for an episode (no training happen)
        success = agent.train(env, episode, 0)

        for i in range(args.n_layers - 1):
            # Get number of goal proposed/achieved
            subgoal_info = agent.subgoal_achieved_info[i]
            subgoal_achieved_ratio = subgoal_info[0] / subgoal_info[1]

            logging.info(f"Subgoal info: {subgoal_info}")
            print(f"Episode: {episode}, Layer {i}, subgoal achieved ratio: {subgoal_achieved_ratio:.2f}")

        test_episode_lengths.append(agent.steps_taken)           

        if success:
            print("Episode %d Length %d Success\n" % (episode, env.steps_cnt))
            successful_test_episodes += 1

    # Log performance
    success_rate = successful_test_episodes / num_test_episodes
    average_episode_length = np.mean(test_episode_lengths)

    print("Step: %d Test Succ. Rate %.2f%%, Avg. Len.: %.2f" % (agent.total_env_steps, success_rate * 100, average_episode_length))
    
    

     
