import joblib
import numpy as np
import pandas as pd   # ← مهم جداً

# تحميل النموذج والبيانات المساعدة
bundle = joblib.load("smart_model_bundle.pkl")

dt_model = bundle["model"]
types_encoder = bundle["types_encoder"]
next_encoder = bundle["next_encoder"]
top4_by_country_solution = bundle["top4_by_country_solution"]

# عرض القيم المسموح بها
available_actions = list(types_encoder.classes_)
available_countries = sorted(set([key[0] for key in top4_by_country_solution.keys()]))
available_solutions = sorted(set([key[1] for key in top4_by_country_solution.keys()]))

print("=== Customer Journey Smart Interface ===\n")
print("Available countries:", available_countries)
print("Available solutions:", available_solutions)
print("Available actions:", available_actions)

# إدخال المستخدم
try:
    user_country = input("\nEnter country_standard (text): ").strip()
    if user_country not in available_countries:
        raise ValueError(f"Invalid country: '{user_country}'")

    user_solution = input("Enter solution (text): ").strip()
    if user_solution not in available_solutions:
        raise ValueError(f"Invalid solution: '{user_solution}'")

    user_action = input("Enter current action (text): ").strip()
    if user_action not in available_actions:
        raise ValueError(f"Invalid action: '{user_action}'")

    user_steps = int(input("How many steps do you want in the trip? "))
    if user_steps <= 0:
        raise ValueError("Number of steps must be positive.")

except Exception as e:
    print("\n❌ Input Error:", e)
    exit()

# دالة بناء الرحلة الذكية
def build_super_smart_trip(model, start_action, country, solution, steps=5):
    trip = []

    current = types_encoder.transform([start_action])[0]
    dynamic_weights = {a: 1 for a in next_encoder.classes_}

    top4_cs = top4_by_country_solution.get((country, solution), {})
    if top4_cs:
        top4_cs = {k: v / max(top4_cs.values()) for k, v in top4_cs.items()}

    last_action = None
    repeat_count = 0

    for _ in range(steps):
        current_action_name = types_encoder.inverse_transform([current])[0]
        trip.append(current_action_name)

        # ← التعديل المهم هنا
        row = pd.DataFrame(
            [[country, solution, current]],
            columns=["country_standard", "solution", "types"]
        )

        probs = model.predict_proba(row)[0]

        dt_weights = {
            next_encoder.inverse_transform([i])[0]: probs[i]
            for i in range(len(probs))
        }

        top4_weights = {a: top4_cs.get(a, 0.1) for a in dt_weights}

        final_weights = {
            a: dt_weights[a] * top4_weights[a] * dynamic_weights[a]
            for a in dt_weights
        }

        sorted_actions = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_actions[:3]

        actions = [a for a, w in top3]
        weights = np.array([w for a, w in top3])
        weights = weights / weights.sum()

        next_action_name = np.random.choice(actions, p=weights)

        if next_action_name == last_action:
            repeat_count += 1
        else:
            repeat_count = 0

        if repeat_count >= 2 and len(actions) > 1:
            next_action_name = actions[1]
            repeat_count = 0

        for a in dynamic_weights:
            if a == next_action_name:
                dynamic_weights[a] *= 0.5
            else:
                dynamic_weights[a] *= 1.05

        current = types_encoder.transform([next_action_name])[0]
        last_action = next_action_name

    return trip

# تنفيذ الرحلة
print("\nGenerating best trip...\n")
trip = build_super_smart_trip(
    dt_model,
    start_action=user_action,
    country=user_country,
    solution=user_solution,
    steps=user_steps
)

print("✅ Best Trip:")
print(" → ".join(trip))
