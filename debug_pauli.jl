using PauliOperators
n=2
p = ScaledPauli{n}(0.5, Pauli(n; Z=[1]))
println("ScaledPauli Type: ", typeof(p))
println("ScaledPauli fields: ", fieldnames(typeof(p)))
println("p.pauli type: ", typeof(p.pauli))
dump(p)
