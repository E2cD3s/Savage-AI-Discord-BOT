from collections import deque
import logging

class MemoryManager:
    def __init__(self, max_history=5):
        self.histories = {} # Dict[channel_id, deque]
        self.histories = {} # Dict[channel_id, deque]
        self.max_history = max_history

    def set_limit(self, new_limit):
        """Update the max history limit and resize existing histories."""
        self.max_history = new_limit
        for channel_id in self.histories:
            # Create new deque with new limit, preserving existing items
            old_deque = self.histories[channel_id]
            self.histories[channel_id] = deque(old_deque, maxlen=new_limit)
        logging.info(f"Memory limit updated to {new_limit}")

    def get_history(self, channel_id):
        if channel_id not in self.histories:
            self.histories[channel_id] = deque(maxlen=self.max_history)
        return list(self.histories[channel_id])

    def add_message(self, channel_id, role, content, username=None):
        if channel_id not in self.histories:
            self.histories[channel_id] = deque(maxlen=self.max_history)
        
        final_content = content
        if username and role == "user":
            final_content = f"[User: {username}] {content}"
            
        self.histories[channel_id].append({"role": role, "content": final_content})
        logging.debug(f"Added message to history for {channel_id}: {role} ({username})")

    def clear_history(self, channel_id):
        if channel_id in self.histories:
            self.histories[channel_id].clear()
            logging.info(f"Cleared history for {channel_id}")

    def clear_all(self):
        """Wipe memory for all channels."""
        for channel_id in self.histories:
            self.histories[channel_id].clear()
        self.histories.clear()
        logging.info("Cleared ALL bot memory.")
