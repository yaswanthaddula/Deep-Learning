#5. Bayes’ Theorem — Spam Email Detection
import matplotlib.pyplot as plt

prior = 0.30
prize_spam = 0.70
prize_not_spam = 0.10

prize = prize_spam * prior + prize_not_spam * (1 - prior)
posterior = prize_spam * prior / prize

print("Prior Probability of Spam:", prior)
print("Posterior Probability of Spam:", round(posterior, 4))

plt.bar(["Prior", "Posterior"], [prior, posterior])
plt.ylabel("Probability")
plt.title("Spam Email Probability")
plt.show()
