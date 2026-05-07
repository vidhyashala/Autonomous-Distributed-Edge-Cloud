package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

type AgentStatus struct {
	Agent  string `json:"agent"`
	Role   string `json:"role"`
	Status string `json:"status"`
}

func main() {
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		status := AgentStatus{Agent: "neurosplit-go-orchestrator", Role: "edge-cloud-policy-agent", Status: "ok"}
		if err := json.NewEncoder(w).Encode(status); err != nil {
			log.Printf("encode health response: %v", err)
		}
	})
	fmt.Println("NeuroSplit-X Go orchestrator listening on :8090")
	log.Fatal(http.ListenAndServe(":8090", nil))
}
