import pystk
import torch
import numpy as np


def control(aim_point, current_vel, target_vel = 25):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    beta = torch.Tensor([1,1,1])
    
    #accel = alpha*f(beta[0]*x[0] + b[0],beta[1]*x[1] + b[1],beta[2]*v + b[2]) + gamma
    #accel = max(0,(target_vel-current_vel)/target_vel)
    accel = target_vel>current_vel
    alpha = .9
    steer_angle = np.arctan(aim_point[0]/aim_point[1])
    drift_angle = np.pi/4
    steer = np.sign(aim_point[0])
    
    if(np.abs(steer_angle) > 1):
        action.drift=True
        action.steer = np.sign(steer)
    else:
        action.drift=False
        action.steer = np.sign(steer)*np.abs(steer_angle)*.95
    
    action.acceleration = accel
    return action

def new_controller(aim_point, current_vel):
    return control(aim_point,current_vel)

def new_action_net():
    return torch.nn.Linear(2*5*3, 1, bias=False)

def rollout_many(many_agents, **kwargs):
    ray_data = []
    for i, agent in enumerate(many_agents):
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
        ray_data.append(how_far)
    return ray_data

class Actor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, track_info, kart_info, **kwargs):
        f = state_features(track_info, kart_info)
        output = self.action_net(torch.as_tensor(f).view(1,-1))[0]

        action = pystk.Action()
        action.acceleration = output[0]
        action.steer = output[1]
        return action
    
def learn_controller(samples = 100, epochs = 5):
    import numpy as np
    pytux = PyTux()
    
    many_action_nets = [new_action_net() for i in range(1000)]
    
    data = rollout_many([Actor(action_net) for action_net in many_action_nets], n_steps=600)

    good_initialization = many_action_nets[ np.argmax([data]) ]
    print(good_initialization)
    
    pytux.close()


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
