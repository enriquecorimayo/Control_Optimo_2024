from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import EvalCallback
from environment.Environment import AvellanedaEnv
from environment.State import s0, T, dt, sigma, beta, k, A, kappa

if __name__ == "__main__":
    env = AvellanedaEnv(s0, T, dt, sigma, beta, k, A, kappa)
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=500,
        verbose=1
    )
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False, callback_after_eval=stop_callback)

    print("Model not found! Starting training...")
    policy_kwargs = dict(net_arch=[10, 10])
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=1.0, tensorboard_log="./logs/")
    total_timesteps = 200000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
