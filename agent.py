import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class ActorCriticAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.gamma = 0.999 

        self.alpha_psi = 0.7 
        self.lambd_psi = 0.7

        self.alpha_theta = 0.7
        self.lambd_theta = 0.7

        self.p = 10
        self.k = 10

        self.e_psi = np.zeros((self.p+1, self.k+1)) #state-value parameterization
        self.e_theta = np.zeros((self.p+1, self.k+1)) #policy parameterization

        self.theta = np.zeros((self.p+1, self.k+1))
        self.sigma = np.ones((self.p+1, self.k+1))*10
        self.psi = np.zeros((self.p+1, self.k+1))

        #self.sigma = 10

        self.I = 1

        self.S = np.zeros((self.p+1,self.k+1))

        self.step = 0
        self.episode = 0

        self.prev_observation = ()
        self.prev_action = 0 
        self.prev_reward = 0


    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """

        self.step = 0
        self.episode += 1

        self.e_psi = np.zeros((self.p+1, self.k+1)) #state-value parameterization
        self.e_theta = np.zeros((self.p+1, self.k+1)) #policy parameterization

        x_min = x_range[0]
        x_max = x_range[1]

        self.Si = np.array([x_min - (i*x_min)/float(self.p) for i in range(0,self.p+1)]).reshape(self.p+1,1)
        self.Sj = np.array([-20 + j*40/float(self.k) for j in range(self.k+1)]).reshape(self.k+1,1)

        self.stepi = (x_max - x_min)/float(self.p)
        self.stepj = 40/float(self.k)


        pass

    def PHI(self, observation):

        return np.dot(np.exp( - ( ( (observation[0] - self.Si)/float(self.stepi) ) **2 ) ),np.transpose(np.exp( - ( ( ( observation[1] - self.Sj)/float(self.stepj) )**2 ) )))

    def V(self, s, w):

        # #print('PHI shape',self.PHI(s).shape)
        # #print('PHI', self.PHI(s))
        # #print('w', w)
        # #print('w shape', w.shape)

        return np.sum(np.copy(np.multiply(w,self.PHI(s))))


    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if self.step == 0:
            action = np.random.normal(0, 10)

        else:
            action = np.random.normal(self.V(observation, self.theta), self.V(observation, self.sigma))/3

        return action

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        if self.step <1:

            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward

        if reward > 0:

            delta = reward - self.V(self.prev_observation, self.psi)

        else:

            delta = reward + self.gamma*self.V(observation, self.psi) - self.V(self.prev_observation, self.psi)
            #print('delta', delta)


        self.e_psi = self.gamma * self.lambd_psi * self.e_psi + self.I * self.PHI(self.prev_observation)
        #print('e_psi', self.e_psi)
        self.e_theta = self.gamma * self.lambd_theta * self.e_theta + self.I * ((self.prev_action - self.V(self.prev_observation, self.theta))*self.PHI(self.prev_observation))/self.V(self.prev_observation, self.sigma)**2 #gradient de ln(policy(A|S,theta))
        #print('e_theta', self.e_theta)

        self.theta += self.alpha_theta * delta * self.e_theta   
        #print('theta', self.theta)
        self.psi += self.alpha_psi * delta * self.e_psi 
        #print('psi', self.psi)
        self.I *= self.gamma

        if self.sigma[0,0] >0.1:
            self.sigma += -0.001

        if self.step >0 :

            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward       

        self.step +=1


        pass




class svgAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.gamma = 0.999 

        self.alpha_psi = 0.8 
        self.lambd_psi = 0.8

        self.alpha_theta = 0.8
        self.lambd_theta = 0.8

        self.p = 10
        self.k = 10

        self.e_psi = np.zeros((self.p+1, self.k+1)) #state-value parameterization
        self.e_theta = np.zeros((self.p+1, self.k+1)) #policy parameterization

        self.theta = np.zeros((self.p+1, self.k+1))
        self.sigma = np.ones((self.p+1, self.k+1))*10
        self.psi = np.zeros((self.p+1, self.k+1))

        #self.sigma = 10

        self.I = 1

        self.S = np.zeros((self.p+1,self.k+1))

        self.step = 0
        self.episode = 0

        self.prev_observation = ()
        self.prev_action = 0 
        self.prev_reward = 0


    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """

        self.step = 0
        self.episode += 1

        self.e_psi = np.zeros((self.p+1, self.k+1)) #state-value parameterization
        self.e_theta = np.zeros((self.p+1, self.k+1)) #policy parameterization

        x_min = x_range[0]
        x_max = x_range[1]

        self.Si = np.array([x_min - (i*x_min)/float(self.p) for i in range(0,self.p+1)]).reshape(self.p+1,1)
        self.Sj = np.array([-20 + j*40/float(self.k) for j in range(self.k+1)]).reshape(self.k+1,1)

        self.stepi = (x_max - x_min)/float(self.p)
        self.stepj = 40/float(self.k)


        pass

    def PHI(self, observation):

        return np.dot(np.exp( - ( ( (observation[0] - self.Si)/float(self.stepi) ) **2 ) ),np.transpose(np.exp( - ( ( ( observation[1] - self.Sj)/float(self.stepj) )**2 ) )))

    def V(self, s, w):

        # #print('PHI shape',self.PHI(s).shape)
        # #print('PHI', self.PHI(s))
        # #print('w', w)
        # #print('w shape', w.shape)

        return np.sum(np.copy(np.multiply(w,self.PHI(s))))


    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if self.step == 0:
            action = np.random.normal(0, 10)

        else:
            action = np.random.normal(self.V(observation, self.theta), self.V(observation, self.sigma))/3

        return action

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        if self.step <1:

            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward

        if reward > 0:

            delta = reward - self.V(self.prev_observation, self.psi)

        else:

            delta = reward + self.gamma*self.V(observation, self.psi) - self.V(self.prev_observation, self.psi)
            #print('delta', delta)


        self.e_psi = self.gamma * self.lambd_psi * self.e_psi + self.I * self.PHI(self.prev_observation)
        #print('e_psi', self.e_psi)
        self.e_theta = self.gamma * self.lambd_theta * self.e_theta + self.I * ((self.prev_action - self.V(self.prev_observation, self.theta))*self.PHI(self.prev_observation))/self.V(self.prev_observation, self.sigma)**2 #gradient de ln(policy(A|S,theta))
        #print('e_theta', self.e_theta)

        self.theta += self.alpha_theta * delta * self.e_theta   
        #print('theta', self.theta)
        self.psi += self.alpha_psi * delta * self.e_psi 
        #print('psi', self.psi)
        self.I *= self.gamma

        if self.sigma[0,0] >0.1:
            self.sigma += -0.001

        if self.step >0 :

            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward       

        self.step +=1


        pass

Agent = svgAgent
