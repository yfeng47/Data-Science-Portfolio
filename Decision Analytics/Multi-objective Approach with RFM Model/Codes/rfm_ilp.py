"""
@authors: 
@date: 20200812
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
m = gp.Model("direct_marketing_ILP")

# Decision variables:
# if this rfm segment should be reached (1, 0)
rfm = m.addVars(rfm_segment, name='rfm', vtype=GRB.BINARY)

# Objective:
# maximize profit
m.setObjective(rfm.prod(
    {key: (purchase_prob[key] * expected_revenue[key] - avg_cost)
           * customer_count[key] for key in rfm_segment}
), GRB.MAXIMIZE)

# Constraints
# Max. budget
m.addConstr(
    (rfm.prod(customer_count) * avg_cost) <= budget, 'budget constraint'
)

# Compute optimal solution
m.optimize()


# Print solution
def print_solution():
    if m.status == GRB.OPTIMAL:
        print('Objective value:', m.objVal)
        m.printAttr(['x'])
        m.printAttr(['Sense', 'Slack', 'RHS'])
    else:
        print('No solution')


# Result
print_solution()
