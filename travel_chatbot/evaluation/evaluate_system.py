# evaluate_system.py
"""
Comprehensive Evaluation Script for Travel ChatBot System
Tests all 6 modules + system-level metrics
Generates detailed performance report
"""

import json
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from collections import defaultdict
import requests

# Import your modules
from ..modules.orchestration import QueryOrchestrator
from ..modules.graph_rag import GraphRAG
from ..modules.crag import CorrectiveRAG
from ..modules.api_module import AskToActAPI
from ..modules.memory import MemoryManager
from ..modules.main_llm import MainLLM

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "http://localhost:8089"  # Your Flask backend
TEST_DATA_DIR = "testsets/"


# ============================================================================
# TEST DATA LOADERS
# ============================================================================

def load_test_data(filename):
    """Load test data from JSON file"""
    try:
        with open(f"{TEST_DATA_DIR}{filename}") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warning: {filename} not found. Skipping tests for this module.")
        return []


# ============================================================================
# MODULE 1: QUERY ORCHESTRATOR EVALUATION
# ============================================================================

def evaluate_orchestrator():
    """Evaluate BERT + LLM classifier"""
    print("\n" + "=" * 70)
    print("MODULE 1: QUERY ORCHESTRATOR EVALUATION")
    print("=" * 70)

    test_data = load_test_data("query_classifier_test.json")
    if not test_data:
        return None

    orch = QueryOrchestrator(bert_model_path="./models/bert_classifier")

    y_true, y_pred, latencies = [], [], []
    label_map = {"Conversational": 0, "RAG": 1, "API_Call": 2}

    for sample in test_data:
        start = time.time()
        pred = orch.classify(sample["text"])
        latency = (time.time() - start) * 1000  # ms

        y_pred.append(label_map[pred])
        y_true.append(sample["label"])
        latencies.append(latency)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    avg_latency = np.mean(latencies)

    print(f"\n✅ Classification Accuracy: {accuracy * 100:.2f}% (Target: ≥95%)")
    print(f"✅ Macro F1-Score: {macro_f1:.3f} (Target: ≥0.92)")
    print(f"✅ Weighted F1-Score: {weighted_f1:.3f} (Target: ≥0.93)")
    print(f"✅ Average Latency: {avg_latency:.1f}ms (Target: <150ms)")

    print(f"\n📊 Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Conversational", "RAG", "API_Call"],
                                digits=3))

    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print("             Pred Conv  Pred RAG  Pred API")
    print(f"True Conv    {cm[0][0]:>9}  {cm[0][1]:>8}  {cm[0][2]:>8}")
    print(f"True RAG     {cm[1][0]:>9}  {cm[1][1]:>8}  {cm[1][2]:>8}")
    print(f"True API     {cm[2][0]:>9}  {cm[2][1]:>8}  {cm[2][2]:>8}")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'avg_latency_ms': avg_latency,
        'confusion_matrix': cm.tolist()
    }


# ============================================================================
# MODULE 2: GRAPH RAG EVALUATION
# ============================================================================

def evaluate_graph_rag():
    """Evaluate entity extraction and retrieval"""
    print("\n" + "=" * 70)
    print("MODULE 2: GRAPH RAG EVALUATION")
    print("=" * 70)

    test_data = load_test_data("graphrag_test.json")
    if not test_data:
        return None

    graphrag = GraphRAG()  # Initialize with your config

    correct_entities, total = 0, 0
    latencies = []

    for sample in test_data:
        start = time.time()
        try:
            extracted = graphrag.extract_entities_with_llm(sample["query"])
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # Check if extracted entities match ground truth
            match = (
                    extracted.get('city', '').lower() == sample.get('city', '').lower()
            )
            correct_entities += match
            total += 1

            if not match:
                print(f"❌ Query: {sample['query']}")
                print(f"   Expected: {sample.get('city')}, Got: {extracted.get('city')}")
        except Exception as e:
            print(f"⚠️  Error processing query: {sample['query']}")
            total += 1

    entity_accuracy = (correct_entities / total) * 100 if total > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print(f"\n✅ Entity Extraction Accuracy: {entity_accuracy:.2f}% (Target: ≥85%)")
    print(f"✅ Average Latency: {avg_latency:.1f}ms (Target: <500ms)")

    return {
        'entity_accuracy': entity_accuracy / 100,
        'avg_latency_ms': avg_latency
    }


# ============================================================================
# MODULE 3: CRAG EVALUATION
# ============================================================================

def evaluate_crag():
    """Evaluate volatility detection"""
    print("\n" + "=" * 70)
    print("MODULE 3: CORRECTIVE RAG (CRAG) EVALUATION")
    print("=" * 70)

    test_data = load_test_data("crag_volatility_test.json")
    if not test_data:
        return None

    crag = CorrectiveRAG()

    tp, fp, tn, fn = 0, 0, 0, 0
    latencies = []

    for sample in test_data:
        start = time.time()
        verdict = crag.should_verify_with_llm(sample["query"], {})
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        pred = "volatile" if verdict else "static"
        true = sample["volatility"]

        if pred == "volatile" and true == "volatile":
            tp += 1
        elif pred == "volatile" and true == "static":
            fp += 1
        elif pred == "static" and true == "static":
            tn += 1
        elif pred == "static" and true == "volatile":
            fn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    avg_latency = np.mean(latencies)

    print(f"\n✅ Volatility Detection Accuracy: {accuracy * 100:.2f}% (Target: ≥88%)")
    print(f"✅ Precision (Volatile): {precision:.3f} (Target: ≥0.85)")
    print(f"✅ Recall (Volatile): {recall:.3f} (Target: ≥0.90)")
    print(f"✅ F1-Score: {f1:.3f}")
    print(f"✅ False Positive Rate: {fpr * 100:.2f}% (Target: ≤12%)")
    print(f"✅ Average Latency: {avg_latency:.1f}ms")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'avg_latency_ms': avg_latency
    }


# ============================================================================
# MODULE 4: API MODULE EVALUATION
# ============================================================================

def evaluate_api_module():
    """Evaluate parameter extraction"""
    print("\n" + "=" * 70)
    print("MODULE 4: ASKTOACT API MODULE EVALUATION")
    print("=" * 70)

    test_data = load_test_data("api_params_test.json")
    if not test_data:
        return None

    api_mod = AskToActAPI()

    correct, total = 0, 0
    latencies = []

    for sample in test_data:
        start = time.time()
        try:
            extracted = api_mod.extract_parameters_with_llm(
                sample["query"],
                sample["api_type"],
                {}
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # Check if all expected params are extracted correctly
            match = all(
                extracted.get(k) == v
                for k, v in sample.get("params", {}).items()
            )
            correct += match
            total += 1

            if not match:
                print(f"❌ Query: {sample['query']}")
                print(f"   Expected: {sample.get('params')}")
                print(f"   Got: {extracted}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            total += 1

    param_accuracy = (correct / total) * 100 if total > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print(f"\n✅ Parameter Extraction Accuracy: {param_accuracy:.2f}% (Target: ≥90%)")
    print(f"✅ Average Latency: {avg_latency:.1f}ms (Target: <500ms)")

    return {
        'param_accuracy': param_accuracy / 100,
        'avg_latency_ms': avg_latency
    }


# ============================================================================
# MODULE 5: MEMORY MANAGER EVALUATION
# ============================================================================

def evaluate_memory():
    """Evaluate STM and LTM"""
    print("\n" + "=" * 70)
    print("MODULE 5: MEMORY MANAGER EVALUATION")
    print("=" * 70)

    test_data = load_test_data("memory_test.json")
    if not test_data:
        return None

    mem = MemoryManager()

    correct_stm, correct_ltm, total = 0, 0, 0

    for convo in test_data:
        mem.reset()
        for turn in convo.get("dialogue", []):
            mem.update(turn)

        # Check STM accuracy
        if mem.get_stm_summary() == convo.get("expected_stm"):
            correct_stm += 1

        # Check LTM accuracy
        if set(mem.get_ltm_preferences()) == set(convo.get("expected_ltm", [])):
            correct_ltm += 1

        total += 1

    stm_accuracy = (correct_stm / total) * 100 if total > 0 else 0
    ltm_accuracy = (correct_ltm / total) * 100 if total > 0 else 0

    print(f"\n✅ STM Context Retention: {stm_accuracy:.2f}% (Target: ≥85%)")
    print(f"✅ LTM Preference Extraction: {ltm_accuracy:.2f}% (Target: ≥80%)")

    return {
        'stm_accuracy': stm_accuracy / 100,
        'ltm_accuracy': ltm_accuracy / 100
    }


# ============================================================================
# MODULE 6: MAIN LLM EVALUATION
# ============================================================================

def evaluate_main_llm():
    """Evaluate response quality with BLEU, ROUGE"""
    print("\n" + "=" * 70)
    print("MODULE 6: MAIN LLM EVALUATION")
    print("=" * 70)

    test_data = load_test_data("response_quality_test.json")
    if not test_data:
        return None

    main_llm = MainLLM(model_name="google/flan-t5-base")

    refs, preds = [], []
    latencies = []

    for sample in test_data:
        start = time.time()
        try:
            pred = main_llm.generate_response(
                sample["query"],
                context=sample.get("context", {})
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            preds.append(pred)
            refs.append(sample["reference"])
        except Exception as e:
            print(f"⚠️  Error: {e}")

    # Calculate BLEU (simplified)
    try:
        from nltk.translate.bleu_score import sentence_bleu
        bleu_scores = [
            sentence_bleu([ref.split()], pred.split())
            for ref, pred in zip(refs, preds)
        ]
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    except:
        avg_bleu = 0
        print("⚠️  NLTK not available for BLEU calculation")

    avg_latency = np.mean(latencies) if latencies else 0

    print(f"\n✅ Average BLEU Score: {avg_bleu:.3f} (Target: ≥0.45)")
    print(f"✅ Average Generation Latency: {avg_latency:.1f}ms (Target: <2000ms)")

    return {
        'bleu_score': avg_bleu,
        'avg_latency_ms': avg_latency
    }


# ============================================================================
# END-TO-END SYSTEM EVALUATION
# ============================================================================

def evaluate_end_to_end():
    """Test complete system via API"""
    print("\n" + "=" * 70)
    print("END-TO-END SYSTEM EVALUATION")
    print("=" * 70)

    test_queries = [
        {"query": "Hello! How are you?", "type": "Conversational"},
        {"query": "What are the best places in Paris?", "type": "RAG"},
        {"query": "What's the weather in Tokyo?", "type": "API_Call"},
    ]

    latencies = []
    successful = 0

    for test in test_queries:
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": test["query"]},
                timeout=10
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            if response.status_code == 200:
                successful += 1
                data = response.json()
                print(f"✅ {test['type']}: {latency:.0f}ms - {data.get('query_type')}")
            else:
                print(f"❌ {test['type']}: Failed (status {response.status_code})")
        except Exception as e:
            print(f"❌ {test['type']}: Error - {e}")

    success_rate = (successful / len(test_queries)) * 100
    avg_latency = np.mean(latencies) if latencies else 0

    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    print(f"✅ Average Response Time: {avg_latency:.1f}ms (Target: <5000ms)")

    return {
        'success_rate': success_rate / 100,
        'avg_latency_ms': avg_latency
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results):
    """Generate comprehensive evaluation report"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)

    report = f"""
# Travel ChatBot System Evaluation Report
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

"""

    # Overall status
    all_pass = True
    critical_metrics = []

    if results.get('orchestrator'):
        orch = results['orchestrator']
        critical_metrics.append(('Classification F1', orch['macro_f1'], 0.92))

    if results.get('end_to_end'):
        e2e = results['end_to_end']
        critical_metrics.append(('Success Rate', e2e['success_rate'], 0.80))
        critical_metrics.append(('Response Time (ms)', e2e['avg_latency_ms'], 5000))

    for metric, value, threshold in critical_metrics:
        status = "✅ PASS" if value >= threshold else "❌ FAIL"
        report += f"- {metric}: {value:.3f} (threshold: {threshold}) {status}\n"
        if value < threshold:
            all_pass = False

    overall_status = "🟢 PASS" if all_pass else "🔴 FAIL"
    report += f"\n**Overall System Status**: {overall_status}\n"

    # Module-specific results
    report += "\n## Module Performance\n\n"

    for module, data in results.items():
        if data:
            report += f"### {module.replace('_', ' ').title()}\n"
            for key, value in data.items():
                if isinstance(value, float):
                    report += f"- {key}: {value:.3f}\n"
                elif isinstance(value, (int, str)):
                    report += f"- {key}: {value}\n"
            report += "\n"

    # Save report
    with open("evaluation_report.md", "w") as f:
        f.write(report)

    print(report)
    print(f"\n📄 Report saved to: evaluation_report.md")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete evaluation suite"""
    print("\n🚀 Starting Comprehensive System Evaluation...")
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    # Run all evaluations
    try:
        results['orchestrator'] = evaluate_orchestrator()
    except Exception as e:
        print(f"❌ Orchestrator evaluation failed: {e}")
        results['orchestrator'] = None

    try:
        results['graph_rag'] = evaluate_graph_rag()
    except Exception as e:
        print(f"❌ Graph RAG evaluation failed: {e}")
        results['graph_rag'] = None

    try:
        results['crag'] = evaluate_crag()
    except Exception as e:
        print(f"❌ CRAG evaluation failed: {e}")
        results['crag'] = None

    try:
        results['api_module'] = evaluate_api_module()
    except Exception as e:
        print(f"❌ API Module evaluation failed: {e}")
        results['api_module'] = None

    try:
        results['memory'] = evaluate_memory()
    except Exception as e:
        print(f"❌ Memory evaluation failed: {e}")
        results['memory'] = None

    try:
        results['main_llm'] = evaluate_main_llm()
    except Exception as e:
        print(f"❌ Main LLM evaluation failed: {e}")
        results['main_llm'] = None

    try:
        results['end_to_end'] = evaluate_end_to_end()
    except Exception as e:
        print(f"❌ End-to-end evaluation failed: {e}")
        results['end_to_end'] = None

    # Generate report
    generate_report(results)

    print(f"\n⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
