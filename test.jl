using SciMLBase

# Proviamo a passare una stringa al posto di una funzione f
# Questo serve per vedere come Julia reagisce a un input sbagliato
f = "non sono una funzione"
u0 = [1.0]
tspan = (0.0, 1.0)

# Questa riga dovrebbe far "arrabbiare" Julia
prob = ODEProblem(f, u0, tspan)