# Structure of MH Algorithm
def model(t, theta):
    alpha, beta, gamma = theta
    # Actual function to try, doesn't work right now because of high complexity
    return alpha * np.exp(t) * (1 - np.tanh(2*(t - beta)) ) * np.sin(gamma * t)
    # return alpha * np.exp(-beta * t) + gamma

def log_likelihood(theta):
    y_model = model(t, theta)
    chi2 = np.sum(((y_data - y_model) / y_err)**2)
    return -0.5 * chi2

def propose(theta, sigma):
    return theta + np.random.normal(0, sigma)  # small step

# Bounded Check
def bounded(theta):
    if (0 < theta[0] < 2) and (1 < theta[1] < 10) and (1 < theta[2] < 20):
        return 1
    return 0

def MHSampling(seed):
    acceptance_count_theta = 0
    total_inbound = 0
    np.random.seed(seed)
    theta = np.array([np.random.randint(1,9)*0.1, np.random.randint(12,98)*0.1, np.random.randint(12, 198)*0.1])
    print("theta: ", theta)
    chain = [theta]
    logL = log_likelihood(theta)
    N = 1000000
    burn_in = N/5
    for i in tqdm(range(N)):
        theta_new = propose(theta, [0.3, 1.5, 3])
        # uniform prior check: only accept if parameters in range
        if bounded(theta_new) == 0:
            if i > burn_in: chain.append(theta)
            continue
        total_inbound += 1
        logL_new = log_likelihood(theta_new)
        A = np.exp(np.minimum(0, logL_new - logL))
        if np.random.rand() < A:
            acceptance_count_theta += 1
            theta, logL = theta_new, logL_new
        if i > burn_in: chain.append(theta)

    chain = np.array(chain)
    acceptance_rate = acceptance_count_theta / total_inbound
    print(f'acceptance rate: {acceptance_rate}')
    return chain