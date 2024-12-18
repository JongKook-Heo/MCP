cd ..
cd ..
cd ..

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_PEBBLE.py env=metaworld_sweep-into-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003  num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=50 num_interact=5000 max_feedback=10000 feed_type=1 reward_update=10 reset_update=100 segment=25 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0.1 teacher_eps_equal=0
done