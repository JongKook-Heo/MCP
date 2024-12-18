cd ..
cd ..
cd ..

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_PEBBLE_mixup.py env=cheetah_run seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=1000000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=100 feed_type=1 reward_update=50 reset_update=100 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0.1
done