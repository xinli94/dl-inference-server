
logs = ['0:start:0', '1:start:2', '2:start:4', '2:end:5', '1:end:6', '0:end:7']
result = [0] * 3
stack = []
for log in logs:
    f, flag, t = log.split(':')
    f = int(f) # function id
    t = int(t) # timestamp

    if flag == 'start':
        if stack:
            result[stack[-1][0]] += (t-1) - stack[-1][1] + 1
        stack.append([f, t])
    else:
        prevF, prevT = stack.pop()
        result[prevF] += t - prevT + 1
        if stack:
            stack[-1][1] = t+1 
print result