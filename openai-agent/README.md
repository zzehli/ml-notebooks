# [orchestrating agents](https://cookbook.openai.com/examples/orchestrating_agents):
```
uv run openai/multi-agents.py 
```
Here is a sample conversation:
```
User: buy stuff
Triage Agent: transfer_to_sales_agent({})
Sales Agent: Are you having any trouble catching those speedy roadrunners?
User: yes
Sales Agent: You should check out our ACME Roadrunner Repellent Gel, it's a game-changer!
User: okay
Sales Agent: execute_order({'price': 9999, 'product': 'ACME Roadrunner Repellent Gel'})


=== Order Summary ===
Product: ACME Roadrunner Repellent Gel
Price: $9999
=================

Confirm order? y/n: y
Order execution successful!
Sales Agent: Your order for ACME Roadrunner Repellent Gel has been placed for a mere $9,999! Just a heads up, it'll only work if you sing lullabies to the roadrunners beforehand.
User: okay
Sales Agent: Your order is confirmed! Enjoy catching those roadrunners!
User: thank you
Sales Agent: You're welcome! If you need anything else, just let me know!
User: i want to see customer service
Sales Agent: transfer_back_to_triage({})
Triage Agent: How can I assist you with customer service today?
User: i need a representative
Triage Agent: escalate_to_human({'summary': 'Customer wants to speak with a representative for assistance.'})
Escalating to human agent...

=== Escalation Report ===
Summary: Customer wants to speak with a representative for assistance.
=========================
```