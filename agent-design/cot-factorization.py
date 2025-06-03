from utils import chat


def main():
    system_message = {"role": "system", "content": """
            
<example>
Question: What are the prime factors of 60?
Step 1: Start with the smallest prime, 2.
60 ÷ 2 = 30

Step 2: Divide 30 by 2 again (since it's still even).
30 ÷ 2 = 15

Step 3: 15 is not divisible by 2. Try the next smallest prime, 3.
15 ÷ 3 = 5

Step 4: 5 is a prime number.

Prime Factorization:
60 = 2 x 2 x 3 x 5
Or, using exponents: 60 = 2^2 x 3 x 5
</example>
<example>

Question: What are the prime factors of 84?
Step 1: Divide by 2.
84 ÷ 2 = 42

Step 2: Divide 42 by 2 again.
42 ÷ 2 = 21

Step 3: 21 is not divisible by 2. Try 3.
21 ÷ 3 = 7

Step 4: 7 is a prime number.

Prime Factorization:
84 = 2 x 2 x 3 x 7
Or, using exponents: 84 = 2^2 x 3 x 7
</example>
"""}
    user_message = {"role": "user", "content": "What are the prime factors of 2299188?"}
    print(chat([system_message, user_message]))


if __name__ == "__main__":
    main()