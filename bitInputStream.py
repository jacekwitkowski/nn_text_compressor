class BitInputStream:
    def __init__(self, f):
        self.f = f
        self.acc = 0
        self.nbits = 0


    def read_bit(self):
        if self.nbits == 0:
            b = self.f.read(1)
            if len(b) == 0:
                return None
            self.acc = b[0]
            self.nbits = 8
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1


    def read_bits(self, n):
        v = 0
        for _ in range(n):
            b = self.read_bit()
            if b is None:
                b = 0
            v = (v << 1) | b
        return v