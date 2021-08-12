from tvm import relay

x = relay.expr.var('x', relay.scalar_type('int64'), dtype = 'int64')
one = relay.expr.const(1, dtype = 'int64')
#help(x)
#help(one)
add = relay.op.tensor.add(x, one)
print(relay.analysis.free_vars(add))
func = relay.Function(relay.analysis.free_vars(add), add, relay.scalar_type('int64'))