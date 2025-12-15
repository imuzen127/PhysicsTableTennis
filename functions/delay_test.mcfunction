# Delay test function
# Tests the delay syntax: N;command

# Immediate (0ms delay)
summon ball 0 1 0 {Tags:['first']}

# 1 second delay
1000;summon ball 0 1.5 0 {Tags:['second']}

# Inherits 1000ms delay (省略ルール)
summon ball 0 2 0 {Tags:['third']}

# 2 second delay from start
2000;summon ball 0 2.5 0 {Tags:['fourth']}
