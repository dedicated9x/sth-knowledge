def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if (n % i) != 0:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        # Case of n is prime number
        factors.append(n)
    return factors

if __name__ == "__main__":
    print('Enter integer:')
    x = input()
    print(f"Prime factors: {prime_factors(int(x))}")