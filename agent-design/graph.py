from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class State:
    """A message passed between nodes in the graph."""

    sender: str
    state_content: Any


class Node:
    """A node in the computation graph that can process messages and produce outputs."""

    def __init__(
        self, node_id: str, 
        process_fn: Callable[[List[State]], Dict[str, Any]],
        is_start: bool = False
    ):
        self.node_id = node_id
        self.process_fn = process_fn
        self.incoming_messages: List[State] = []
        self.is_start = is_start

    def receive_message(self, message: State):
        """Add a message to the node's incoming message queue."""
        self.incoming_messages.append(message)

    def process_messages(self) -> Dict[str, Any]:
        """Process all incoming messages and produce outputs."""
        if self.is_start:
            self.is_start = False
            return self.process_fn([])
        if not self.incoming_messages:
            return {}

        # Process messages using the node's processing function
        outputs = self.process_fn(self.incoming_messages)

        # Clear processed messages
        self.incoming_messages = []

        return outputs


class Graph:
    """A computation graph that manages message passing between nodes."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.conditional_edges: Dict[str, Any] = {}
        self.step_count = 0
        self.output_trace = []

    def add_node(self, node: Node):
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, from_node: str, to_node: str):
        """Add a directed edge between nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")
        self.edges[from_node].append(to_node)
    
    def add_conditional_edge(
        self,
        from_node: str,
        condition_fn: Callable[[Dict[str, Any]], str],
        to_nodes: Dict[str, str],
    ):
        """
        Add a conditional edge that routes messages based on a condition.

        Args:
            from_node: The node that will be the source of the conditional edge.
            condition: A function that takes the output of the from_node and returns a string output.
            to_nodes: A dictionary of function output to node ids.
        """
        if from_node not in self.nodes:
            raise ValueError("From node must exist in the graph")
        for to_node in to_nodes.values():
            if to_node not in self.nodes:
                raise ValueError(f"To node {to_node} must exist in the graph")

        self.conditional_edges[from_node] = {
            "condition_fn": condition_fn,
            "targets": to_nodes,
        }

    def step(self) -> bool:
        """Execute one superstep of the computation.
        Returns True if any messages were processed, False if the computation is complete.
        """
        self.step_count += 1
        messages_processed = False

        # Process all nodes and collect their outputs
        node_outputs = {}
        for node_id, node in self.nodes.items():
            outputs = node.process_messages()
            if outputs:
                messages_processed = True
                node_outputs[node_id] = outputs
                self.output_trace.append({node_id: outputs})
                # print(f"message from {node_id}: {outputs}")

        # Distribute messages to target nodes
        for source_id, outputs in node_outputs.items():
            if source_id in self.conditional_edges:
                # Handle conditional routing
                conditional_edge = self.conditional_edges[source_id]
                condition_fn = conditional_edge["condition_fn"]
                target_map = conditional_edge["targets"]
                target_id = target_map[condition_fn(outputs)]

                message = State(sender=source_id, state_content=outputs)
                self.nodes[target_id].receive_message(message)

            elif source_id in self.edges:
                # Handle standard routing
                for target_id in self.edges[source_id]:
                    message = State(sender=source_id, state_content=outputs)
                    self.nodes[target_id].receive_message(message)

        return messages_processed

    def get_output_trace(self) -> Dict[str, Any]:
        """Get all outputs from all nodes in the graph."""
        return self.output_trace

    def run(self, max_steps=50) -> Dict[str, Any]:
        """Run graph and return output trace"""
        self.step_count = 0
        while self.step_count < max_steps:
            if not self.step():
                break
        return self.get_output_trace()

# Example usage
def create_example_graph():
    # Create nodes with simple processing functions
    def double_numbers(messages: List[State]) -> Dict[str, Any]:
        values = [msg.state_content["value"] for msg in messages]
        return {"value": sum(values) * 2}

    def add_one(messages: List[State]) -> Dict[str, Any]:
        values = [msg.state_content["value"] for msg in messages]
        return {"value": sum(values) + 1}

    def to_node2(input: Dict[str, Any]) -> str:
        values = input["value"]
        if values // 3 == 0:
            return "node2"
        else:
            return "node3"

    # Create graph
    graph = Graph()

    # Add nodes
    node1 = Node("node1", double_numbers)
    node2 = Node("node2", add_one)
    node3 = Node("node3", double_numbers)

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Add edges
    graph.add_conditional_edge("node1", to_node2, {"node2": "node2", "node3": "node3"})
    graph.add_edge("node2", "node3")

    # Initialize with a message
    node1.receive_message(State("initial", {"value": 5}))

    return graph


if __name__ == "__main__":
    # Create and run example graph
    graph = create_example_graph()

    # Run until no more messages are being processed
    res = graph.run()
    print("==================run trace====================")
    for m in res:
        print(m)
