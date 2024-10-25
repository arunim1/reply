Goal: evaluate the assistant-response as an attack vector for jailbreaking 

Existing jailbreaking methods only allow the attack to be on the user portions of the chat. (e.g. PAIR, GCG) 

For PAIR, the change is relatively simple. Prompt the attacking LLM to generate both an {n+1} attack prompts and {n} prior chat responses. Arrange these into a conversation, and then prompt the assistant with the conversation.

For GCG, the change is also relatively simple, but I'll need to be thoughtful and recall how this all works. 
One option is to input {malicious prompt + suffix1} {suffix2} {suffix3} and observe how this changes things, if at all. This has some issues though, as it's not quite multi-turn in some ways, or it feels like it's not priming the target model in the right way. 
The other option is to input {CONST (e.g. Hello)} {longer suffix1} {malicious prompt}  or {CONST (e.g. Hello) suffix1} {suffix2} {malicious prompt} or {CONST (e.g. Hello)} {suffix1} {malicious prompt + suffix2} and observe how this changes things, if at all. 

TODO: 
- [ ] Proof-of-concept using PAIR. goal: get the attack to work on the assistant portion of the chat and measure the change in success rate.  
- [ ] GCG implementation for each fo the options above, measure the change in success rate. 