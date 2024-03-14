
class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_height_obs,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        self.num_height_obs = num_height_obs

        activation = get_activation(AC_Args.activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)


        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history + self.num_height_obs, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history + self.num_height_obs, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history, obs_height):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent, obs_height), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, obs_height, **kwargs):
        self.update_distribution(observation_history, obs_height)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"], ob["obs_height"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], ob["obs_height"], policy_info=policy_info)

    def act_student(self, observation_history, obs_height, policy_info={}):
        latent = self.adaptation_module(observation_history)       
        actions_mean = self.actor_body(torch.cat((observation_history, latent, obs_height), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, obs_height, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info, obs_height), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, obs_height, **kwargs):
        value = self.critic_body(torch.cat((observation_history, privileged_observations, obs_height), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)