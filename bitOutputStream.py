class BitOutputStream:
    def __init__(self, f):
        self.f = f
        self.acc = 0
        self.nbits = 0


    def write_bit(self, b):
        self.acc = (self.acc << 1) | (b & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.f.write(bytes([self.acc]))
            self.acc = 0
            self.nbits = 0  

    def write_bits(self, bits, n):
        for i in reversed(range(n)):
            self.write_bit((bits >> i) & 1)


    def flush(self):
        if self.nbits > 0:
            self.acc = self.acc << (8 - self.nbits)
            self.f.write(bytes([self.acc]))
            self.acc = 0
            self.nbits = 0