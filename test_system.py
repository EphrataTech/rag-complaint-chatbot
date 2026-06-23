"""
Comprehensive Test Suite for CrediTrust Financial RAG System
Validates end-to-end functionality from data to deployment
"""

import sys
import os
from pathlib import Path
import traceback
import time
import json

# Add src to path
sys.path.append('src')

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss
        print("  ✅ Core ML libraries imported successfully")
        
        from task1_eda_preprocessing import main as task1_main
        from task2_embedding_pipeline import ComplaintVectorStore
        from task3_rag_pipeline import RAGPipeline, RAGEvaluator
        print("  ✅ Custom modules imported successfully")
        
        import gradio as gr
        print("  ✅ UI libraries imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_data_preprocessing():
    """Test Task 1 functionality"""
    print("\n🧪 Testing Task 1 - Data Preprocessing...")
    
    # Check if processed data exists
    processed_file = Path('data/processed/filtered_complaints.csv')
    
    if processed_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(processed_file)
            print(f"  ✅ Processed data loaded: {len(df)} complaints")
            
            # Check required columns
            required_cols = ['product_category', 'Consumer complaint narrative']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ⚠️ Missing columns: {missing_cols}")
                return False
            
            print(f"  ✅ Data validation passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Error loading processed data: {e}")
            return False
    else:
        print("  ⚠️ Processed data not found. Run Task 1 first.")
        return False

def test_vector_store():
    """Test Task 2 functionality"""
    print("\n🧪 Testing Task 2 - Vector Store...")
    
    vector_store_dir = Path('vector_store')
    faiss_index_file = vector_store_dir / 'faiss_index.index'
    metadata_file = vector_store_dir / 'faiss_metadata.pkl'
    
    if faiss_index_file.exists() and metadata_file.exists():
        try:
            import faiss
            import pickle
            
            # Load FAISS index
            index = faiss.read_index(str(faiss_index_file))
            print(f"  ✅ FAISS index loaded: {index.ntotal} vectors")
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            print(f"  ✅ Metadata loaded: {len(metadata)} chunks")
            
            # Test embedding model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            test_embedding = model.encode(['test query'])
            print(f"  ✅ Embedding model works: {test_embedding.shape}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error loading vector store: {e}")
            return False
    else:
        print("  ⚠️ Vector store not found. Run Task 2 first.")
        return False

def test_rag_pipeline():
    """Test Task 3 functionality"""
    print("\n🧪 Testing Task 3 - RAG Pipeline...")
    
    try:
        from task3_rag_pipeline import RAGPipeline
        
        # Initialize pipeline
        rag = RAGPipeline(vector_store_path=Path('vector_store'))
        
        if not rag.faiss_index:
            print("  ⚠️ RAG pipeline not initialized. Vector store missing.")
            return False
        
        # Test query
        test_query = "What are credit card billing issues?"
        response = rag.run_rag_pipeline(test_query, k=3)
        
        print(f"  ✅ RAG pipeline response generated")
        print(f"  📊 Query: {response['query']}")
        print(f"  📊 Answer length: {len(response['answer'])} chars")
        print(f"  📊 Sources: {response['num_sources']}")
        
        # Test evaluation
        from task3_rag_pipeline import RAGEvaluator
        evaluator = RAGEvaluator(rag)
        
        # Quick evaluation with 2 questions
        test_questions = [
            "What are credit card problems?",
            "Why do loans get denied?"
        ]
        
        print(f"  🔍 Running quick evaluation...")
        quick_results = []
        for question in test_questions:
            resp = rag.run_rag_pipeline(question, k=3)
            quality = evaluator.evaluate_response_quality(question, resp)
            quick_results.append(quality['quality_score'])
        
        avg_quality = sum(quick_results) / len(quick_results)
        print(f"  ✅ Evaluation completed. Avg quality: {avg_quality:.2f}/5.0")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RAG pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_interface():
    """Test Task 4 functionality"""
    print("\n🧪 Testing Task 4 - Interface Components...")
    
    try:
        # Test Gradio import and basic functionality
        import gradio as gr
        print("  ✅ Gradio imported successfully")
        
        # Test interface class
        sys.path.append('.')
        from app import ComplaintChatInterface
        
        chat_interface = ComplaintChatInterface()
        print("  ✅ Chat interface initialized")
        
        # Test sample questions
        sample_questions = chat_interface.get_sample_questions()
        print(f"  ✅ Sample questions loaded: {len(sample_questions)}")
        
        # Test query processing (mock)
        if chat_interface.rag_pipeline and chat_interface.rag_pipeline.faiss_index:
            history, sources, json_sources = chat_interface.process_query(
                "Test query for interface", 
                []
            )
            print("  ✅ Query processing works")
        else:
            print("  ⚠️ Interface query processing skipped (no vector store)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Interface test failed: {e}")
        return False

def test_end_to_end():
    """Test complete end-to-end workflow"""
    print("\n🧪 Testing End-to-End Workflow...")
    
    try:
        # Test complete pipeline
        from task3_rag_pipeline import RAGPipeline
        
        rag = RAGPipeline()
        
        if not rag.faiss_index:
            print("  ⚠️ End-to-end test skipped (no vector store)")
            return False
        
        # Simulate user interaction
        business_questions = [
            "What are the top credit card complaints?",
            "Why do customers have loan issues?",
            "What money transfer problems exist?"
        ]
        
        print("  🎯 Testing business use cases...")
        
        for i, question in enumerate(business_questions, 1):
            start_time = time.time()
            response = rag.run_rag_pipeline(question, k=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            print(f"    {i}. Query: {question}")
            print(f"       Response time: {response_time:.2f}s")
            print(f"       Sources found: {response['num_sources']}")
            print(f"       Answer preview: {response['answer'][:100]}...")
            print()
        
        print("  ✅ End-to-end workflow successful")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end test failed: {e}")
        return False

def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("🎯 CREDITRUST RAG SYSTEM - COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    tests = [
        ("Imports & Dependencies", test_imports),
        ("Data Preprocessing (Task 1)", test_data_preprocessing), 
        ("Vector Store (Task 2)", test_vector_store),
        ("RAG Pipeline (Task 3)", test_rag_pipeline),
        ("Interface (Task 4)", test_interface),
        ("End-to-End Workflow", test_end_to_end)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "⚠️ PARTIAL"
        except Exception as e:
            results[test_name] = False
            status = "❌ FAILED"
            print(f"  ❌ Unexpected error: {e}")
        
        print(f"\n{test_name}: {status}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print()
    
    for test_name, result in results.items():
        status_emoji = "✅" if result else "❌"
        print(f"{status_emoji} {test_name}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if not results.get("Data Preprocessing (Task 1)", False):
        print("📝 1. Run Task 1: python src/task1_eda_preprocessing.py")
    
    if not results.get("Vector Store (Task 2)", False):
        print("📝 2. Run Task 2: python src/task2_embedding_pipeline.py")
    
    if all(results.values()):
        print("🎉 All systems operational! Ready for deployment.")
        print("🚀 Launch interface: python app.py --interface gradio")
    
    # Save report
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_results': results,
        'summary': {
            'passed': passed,
            'total': total,
            'success_rate': passed/total*100
        }
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/test_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📋 Detailed report saved to: reports/test_report.json")

def main():
    """Main test execution"""
    print("🏦 CrediTrust Financial RAG System - Test Suite")
    print("=" * 60)
    
    generate_test_report()

if __name__ == "__main__":
    main()