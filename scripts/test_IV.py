import pymc as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

gamma_data = np.random.gamma(2, 0.5, size=200)
#sns.histplot(gamma_data)

with pm.Model() as gamma_model:
    alpha = pm.Exponential("alpha", 0.1)
    beta = pm.Exponential("beta", 0.1)

    y = pm.Gamma("y", alpha, beta, observed=gamma_data)

with gamma_model:
    #mean_field = pm.fit()
    mean_field = pm.fit(obj_optimizer=pm.adagrad_window(learning_rate=1e-2))

#plt.plot(mean_field.hist)

with gamma_model:
    trace = pm.sample()

approx_sample = mean_field.sample(1000)

sns.kdeplot(trace.posterior["alpha"].values.flatten(), label="NUTS")
sns.kdeplot(approx_sample.posterior["alpha"].values.flatten(), label="ADVI")
plt.legend()
plt.show()