import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
});

export const chatQuery = async (query) => {
  const res = await API.post("/ask", { query });
  return res.data;
};

export const getChatResponse = async (query) => {
  const res = await API.post("/chat", { query });
  return res.data;
};

export const verifyText = async (text) => {
  const res = await API.post("/verify", { text });
  return res.data;
};
