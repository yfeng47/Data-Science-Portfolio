"""
@authors: 
@date: 20200816
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# Base data
df = pd.read_csv('./model_data.csv')

"""
    R  F  M  CustomerCount  ExpectedRevenue  Probability
0   4  1  1             69       296.609596     0.347826
1   3  1  1            146       296.609596     0.171233
2   2  1  1            193       296.609596     0.145078
3   1  1  1            359       296.609596     0.061281
4   4  1  2             41       308.125397     0.439024
"""

customer_count = {(df['R'][i], df['F'][i], df['M'][i]) : 
                   df['CustomerCount'][i] for i in range(len(df))}
expected_revenue = {(df['R'][i], df['F'][i], df['M'][i]) :
                     df['ExpectedRevenue'][i] for i in range(len(df))}
purchase_prob = {(df['R'][i], df['F'][i], df['M'][i]) : 
                  df['Probability'][i] for i in range(len(df))}
rfm_segment = list(customer_count.keys())

# Making up additional data
avg_cost = 3.5
budget = 7000

# Model
m = gp.Model("direct_marketing_GP")

# Decision variables:
# if this rfm segment should be reached (1, 0)
rfm = m.addVars(rfm_segment, name='rfm', vtype=GRB.BINARY)

# Objective:
# maximize profit
m.ModelSense = GRB.MAXIMIZE

# Goal 1: Keep only R with 1 or 2 (weight: 200)
m.setObjectiveN(rfm.prod(
    {key: (purchase_prob[key] * expected_revenue[key]  - avg_cost)
      * customer_count[key] if key[0] in (1, 2) else 0
      for key in rfm_segment}),
    0,  weight=200, name='prioritize_low_r')

# Goal 2: Keep only F with 3 or 4 (weight: 100)
m.setObjectiveN(rfm.prod(
    {key: (purchase_prob[key] * expected_revenue[key] - avg_cost) 
      * customer_count[key] if key[1] in (3, 4) else 0
      for key in rfm_segment}),
    1,  weight=100, name='prioritize_high_f')

# Goal 3: Keep only M with 3 or 4 (weight: 50)
m.setObjectiveN(rfm.prod(
    {key: (purchase_prob[key] * expected_revenue[key] - avg_cost) 
      * customer_count[key] if key[2] in (3, 4) else 0
      for key in rfm_segment}),
    2,  weight=50, name='prioritize_high_m')

# Constraints
# Max. budget
m.addConstr(
    (rfm.prod(customer_count) * avg_cost) <= budget, 'budget constraint'
)

# Compute optimal solution
m.optimize()


#Print solution
def print_solution():
    if m.status == GRB.OPTIMAL:
        decision = m.getAttr('x')
        print('Objective value: ',
             (df['CustomerCount'] * df['ExpectedRevenue'] * df['Probability'])
              @ decision)
        m.printAttr(['x'])
        m.printAttr(['Sense', 'Slack', 'RHS'])
    else:
        print('No solution')


# Result
print_solution()
