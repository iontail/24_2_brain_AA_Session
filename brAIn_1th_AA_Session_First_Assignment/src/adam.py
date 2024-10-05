import numpy as np

class Adam:
    """
    numpy로 구현한 Adam 옵티마이저
    """
    def __init__(self, alpha: float = 0.001,
                       beta1: float = 0.99,
                       beta2: float = 0.999,
                       eps: float = 1e-8) -> None:
        self.velocity = dict()
        self.momentum = dict()
        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.t = 1 # 스텝
        
        """
        M(t) = b1 * M(t-1) + (1-b1) * cost
        V(t) = b2 * V(t-1) + (1-b2) * (cost**2)
        
        M_hat(t) = M(t) / (1 - (b1**t))
        V_hat(t) = V(t) / (1 - (b2**t))
        
        W(t+1) = W(t) - LR * M_hat / sqrt(V_hat + eps)
        
        """
    def step(self):
        self.t = self.t + 1
    
    def update_grad(self, layer_name:str, layer, LR:float) -> None:
        
        """
        Args:
            layer_name (str): 레이어 이름
            layer (_type_): 레이어
            LR (float): 학습률
        """
        # 현재 레이어의 속도/모멘텀 저장
        self.save_velocity(layer_name, layer)
        self.save_momentum(layer_name, layer)

        # 레이어 그래디언트 업데이트
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            name = (layer_name + grad_key)

            #################### 여기에서 수정하세요 ####################
            m_hat = self.momentum[name] / (1 - (self.beta1 ** self.t))
            v_hat = self.velocity[name] / (1 - (self.beta2 ** self.t))
            
            layer.params[param_key] = layer.params[param_key] - (self.alpha * m_hat / np.sqrt(v_hat + self.eps))
            ###################################################

    def save_velocity(self, layer_name, layer):
        """
        현재 레이어의 속도 저장
        """
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            name = (layer_name + grad_key)
            
            #################### 여기에서 수정하세요 ####################
            if name in self.velocity.keys():
                velocity = self.beta2 * self.velocity[name] + (1 - self.beta2) * (layer.grads[grad_key]**2)
            else:
                velocity = (1 - self.beta2) * (layer.grads[grad_key]**2)
            self.velocity[name] = velocity

            ###################################################

    
    def save_momentum(self, layer_name, layer):
        """
        현재 레이어의 모멘텀 저장
        """
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            name = (layer_name + grad_key)

            #################### 여기에서 수정하세요 ####################
            if name in self.momentum.keys():
                momentum = self.beta1 * self.momentum[name]+ (1 - self.beta1) * layer.grads[grad_key]
            else:
                momentum = (1 - self.beta1) * layer.grads[grad_key]
            self.momentum[name] = momentum
            ###################################################