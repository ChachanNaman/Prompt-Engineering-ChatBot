import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the chat title and input", () => {
  render(<App />);
  expect(screen.getByText(/Prompt Engineering Chatbot/i)).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/Type your message/i)).toBeInTheDocument();
});

test("send button is disabled with empty input", () => {
  render(<App />);
  const sendButton = screen.getByRole("button", { name: /send/i });
  expect(sendButton).toBeDisabled();
});
