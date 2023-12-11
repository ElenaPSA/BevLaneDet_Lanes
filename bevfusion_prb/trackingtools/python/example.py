import TrackingUtilsPy
import numpy as np
obs=np.asarray([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
filter=TrackingUtilsPy.Filter(obs)
state=filter.getState()
print(state)
probs=filter.getModelProbabilities()
print(probs)
filter.predict(0.1)
state=filter.getState()
print(state)
obs[0]=10.0
filter.update(obs)

state=filter.getState()
probs=filter.getModelProbabilities()
print(probs)