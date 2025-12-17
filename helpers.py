
def probs_to_cumfreq(probs, total=1 << 16):
# probs: numpy-like list or tensor of length 256 with non-negative floats summing ~1
# returns cumulative freq array length 257 (cum[0]=0, cum[256]=total)
    p = probs.cpu().numpy() if hasattr(probs, 'cpu') else probs
    p = p.astype(float)
    p = p / (p.sum() + 1e-12)
    freqs = (p * (total - 256)).astype(int) # reserve at least 1 count per symbol
    # ensure at least 1
    for i in range(len(freqs)):
        if freqs[i] <= 0:
           freqs[i] = 1
        # scale to total
    s = freqs.sum()
    # adjust to match total
    while s > total:
    # decrement largest
        i = freqs.argmax()
        if freqs[i] > 1:
            freqs[i] -= 1
            s -= 1
        else:
            break
    while s < total:
        i = freqs.argmax()
        freqs[i] += 1
        s += 1
    cum = [0]
    acc = 0
    for f in freqs:
        acc += int(f)
        cum.append(acc)
    return cum