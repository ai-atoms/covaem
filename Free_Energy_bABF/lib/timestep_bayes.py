import numpy as np

def pbc_diff(v,Cell):
    return v-np.dot(Cell,np.floor(np.dot(np.linalg.inv(Cell),v.T)+0.5)).T # make sure no cross-boundary jumps.


class integrator:
    def __init__(self,lmp,Cell,_x,_v,_f,dt=0.001,gamma=.004,mass=50.,temperature=100.,seed=None,block_rad=10.0,cosmin_mask=None,external_stress_data=None, cv=None,spring=0.): 
        if not seed is None:
            np.random.seed(seed=seed)
        else:
            np.random.seed(seed=int(np.random.uniform(1000)))
        self.lmp = lmp
        self.Cell = Cell

        self.dt = dt
        self.gamma = gamma
        self.mass = mass
        self.heat = np.sqrt(mass * gamma * 1.38e-4 * temperature / 1.6 * dt)

        self.x = _x
        self.v = _v
        self.f = _f

        self.sh = _x.shape

        self.ph = np.random.normal(0.,1.,size=self.sh) * self.heat
        self.ms = 0.
        self.defect_force = 0.

        self.block_rad = block_rad
        self.cosmin_mask = cosmin_mask

        self.deviate_flag = False
        self.external_stress_data=external_stress_data

        self.spring = spring

    def max_step(self):
        return self.ms

    def brownian(self):
        _heat = np.random.normal(0.,1.,size=self.sh) * self.heat
        self.lmp.command('run 1 pre no post no')
        _step = _heat + self.ph + self.f * self.dt #brownian step (no momentum, random heat + force*dt) ?
        _step -= _step.mean(axis=0) #remove translations
        _step /= (self.mass * self.gamma) #scale step
        self.x[:] += _step
        self.ph = _heat.copy()
        self.ms = abs(_step.flatten()).max()
        return self.x,self.f

    def eABF_brownian(self,depsi,biasf):
        _heat = np.random.normal(0.,1.,size=self.sh) * self.heat
        self.lmp.command('run 1 pre no post no')
        _step = _heat + self.ph + self.f * self.dt + biasf *depsi* self.dt #brownian step
        _step -= _step.mean(axis=0) #remove translations
        _step /= (self.mass * self.gamma) #scale step
        self.x[:] += _step
        self.ph = _heat.copy()
        self.ms = abs(_step.flatten()).max()
        return self.x,self.f
