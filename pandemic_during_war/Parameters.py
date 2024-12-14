parameters = {
   'POPULATION_SIZE': (10**6, 10**8),  # Initial population size (1,000,000 - 100,000,000) int(1e2) 100 int(1e6) 10000
    'WAR_ZONE_NODES_COUNT': (1,20),  # Total numbers of warzones (1-20) 2#TODO: set to a fixed size 10
    'CIVILIAN_ZONE_NODES_COUNT': (10,100),   # Total numbers of civilian zones (10-100) 4 #TODO: set to a fixed size 40
    'HOSPITAL_NODES_COUNT': (1,6),
    'SOLDIER_PORTION': (0.01, 0.1),  # the portion of soldiers from the population (0.01 − 0.1), 0.05
    'COMMUTE_RATE_C': (10**-5, 5*10**-4),   # Average commute rate between locations civilian (0.00001 to 0.0005) 2e-4
    'COMMUTE_RATE_W': (10**-5, 5*10**-4),  # Average commute rate between locations civilian (0.00001 to 0.0005) 3e-4
    'TIME_STEP': 1, # Time steps in each simulation iteration (Δt), in hours
    'BETA':  (0.0014, 0.0072),  # Average exposure rate (β, 0.0014 − 0.0072) 0.1
    'RHO': (0.01, 0.01) ,  # Symptomatic rate (ρ, 0.01 − 0.01)
    'LAMBDA': (0.10, 0.10),  # Symptomatic death rate (λ, 1 - GAMMA_S)
    'PSI': (4, 7), # ψ Average incubation time in days spent in the exposed state 4 − 7 [73]
    'GAMMA_A': (7, 10),  # γa Average time in days spent in the Asymptomatic infected state 7 − 10
    'GAMMA_S': (10, 18),  # γs Average time in days spent in the Symptomatically infected state 10 − 18 
    'HOSPITAL_CAPACITY_RANGE': (1000, 10000),  # Maximum number of patients hospitals can treat without effect on clinical performance (μs) 5000 200
    'OVERFLOW_PERCENT': (1.25, 1.5),  # Maximum number of patients hospitals can treat over μs (μf), in percentage 125% − 150%  1.1
    'BASELINE_RECOVERY_RATE': (0.95, 0.99),  # The baseline rate of recovery from pandemic or war injuries (ζs,  0.95 − 0.99) 
    'OVERFLOW_RECOVERY_RATE': (0.8, 0.95),  # The baseline rate of recovery from pandemic or war injuries when the hospital is full (ζf, 0.8 − 0.95)
    'INFECTED_PROPORTION' : 0.10, #TODO: change to 0.1 and split half to sym and asym
    'd': 4.07e-5,   # Death rate of wounded soldiers if not treated due to war 0.0000407
    'epsilon_a': (10**-5,5*10**-5), # Military death rate due to war Lanchester model  0.00001 to 0.00005
    'epsilon_b': (10**-5,5*10**-5),  # Military death rate due to war Lanchester model 0.00001 to 0.00005
    'theta_a': (10**-5,5*10**-5),  # Military injury rate due to war Lanchester model 0.00001 to 0.00005
    'theta_b': (10**-5,5*10**-5),  # Military injury rate due to war Lanchester model 0.00001 to 0.00005
    'NUM_LAYERS': 2,
    'HIDDEN_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'DISCOUNT_FACTOR': 0.98,
    'EXPLORATION_RATE': 0.15,
    'EXPLORATION_DEDAY_RATE': 0.985, 
    'MIN_EXPLORATION_RATE': 0.02,
    'REPLAY_BUFFER_SIZE': 1000,  ## 104
    'TARGET_UPDATE_FREQ': 1000, ## 1000
    'EPISODES_SIZE': 21600,  ##TODO: old is 2160
    'BATCH_SIZE': 32, ## 3
    'SEQUENCE_LENGTH': 32 ##12
}