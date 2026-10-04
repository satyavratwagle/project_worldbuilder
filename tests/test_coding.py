import string

# Define the Base62 alphabet (0-9, a-z, A-Z)
ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase

def encode_base62(text: str) -> str:
    # Convert text to bytes, then to a massive integer
    data = text.encode("utf-8")
    num = int.from_bytes(data, 'big')
    if num == 0:
        return ALPHABET[0]
    
    res = []
    while num > 0:
        num, rem = divmod(num, 62)
        res.append(ALPHABET[rem])
    return ''.join(reversed(res))

def decode_base62(code: str) -> str:
    num = 0
    for char in code:
        num = num * 62 + ALPHABET.index(char)
    
    # Convert integer back to bytes, then to string
    byte_length = (num.bit_length() + 7) // 8
    data = num.to_bytes(byte_length, 'big')
    return data.decode("utf-8")


base_string = "topic_1"

code = encode_base62(base_string)
recovered_string = decode_base62(code)

print(f"{base_string} --> {code} --> {recovered_string}")
