#4. Bayes’ Theorem — Disease Detection
import matplotlib.pyplot as plt

prior = 0.02
sensitivity = 0.95
false_positive = 0.05

positive = sensitivity * prior + false_positive * (1 - prior)
posterior = sensitivity * prior / positive

print("Prior Probability:", prior)
print("Posterior Probability:", round(posterior, 4))

plt.bar(["Prior", "Posterior"], [prior, posterior])
plt.ylabel("Probability")
plt.title("Disease Probability")
plt.show()
