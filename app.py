from flask import Flask, request, jsonify
import json, os
import rag_milvus
import uuid
import shutil
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Milvus
app = Flask(__name__)

@app.route('/create_collection', methods=['POST'])
def process_createCollection():
    try:
        data = request.get_json() 
        if data:                       
            vector_store = rag_milvus.create_collection(data)
            print(f"Collection {data['name']} created ")
            return {"Vector store collection created with collectionName": data['name']}
        else:
            return jsonify({"error": "No JSON payload received"}), 400 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ingest_data', methods=['POST'])
def process_ingestData():
    uuid_str = str(uuid.uuid4())
    try:
        data = request.get_json()  
        if data:
            vectore_store = rag_milvus.create_collection(data)
            if data["url"] != "":
                docs = rag_milvus.extract_web_data(data)                
                documents = rag_milvus.do_chunking(docs,data["chunkingMethod"], data["chunk_size"],data["chunk_overlap"])
            else:                            
                rag_milvus.base64_to_file(data["file_base64_string"],data["fileName"],uuid_str)
                print(f"{data['file_name']} loading....")                     
                documents = rag_milvus.convert_to_documents(uuid_str, data["fileName"], data["chunkingMethod"], data["chunk_size"],data["chunk_overlap"])            
                try:                    
                    Proj_path=os.environ["PROJ_PATH"]    
                    shutil.rmtree(Proj_path+uuid_str)
                    print(f"Directory '{Proj_path+uuid_str}' and all its contents have been deleted successfully.")
                except Exception as e:
                    print(f"Error: {e}")
                
            rag_milvus.insert_data(vectore_store, documents)
            return f"document inserted into vector store collection. CollectionName: {data['name']}, documentId: {uuid_str}"           
        else:
            return jsonify({"error": "No JSON payload received"}), 400        
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

   
@app.route('/delete_document',methods=['DELETE'])
def process_delete_documents():
    try:
        data = request.get_json()        
        if data:
            status = rag_milvus.delete_document(data['doc_id'],data['collection_name'])
            return status
        else:
            return jsonify({"error": "No JSON payload received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500      


@app.route('/get_retriever',methods=['POST'])
def process_get_retriever():
    try:
        data = request.get_json()        
        if data:            
            retriever = rag_milvus.get_retriever(data)
            return retriever
        else:
            return jsonify({"error": "No JSON payload received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500   

@app.route('/web_scrapper',methods=['POST'])
def process_web_scrapper():
    try:
        data = request.get_json()        
        if data:            
            retriever = rag_milvus.extract_web_data(data)
            return retriever
        else:
            return jsonify({"error": "No JSON payload received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500   

# @app.route('/ocr', methods=['POST'])
# def process_request_ocr():
#     try:
#         data = request.get_json()
#         #marks = json.loads(data)  # Parse JSON string to dictionary
#         output = main(data["inpath"],data["outpath"])             
#         if not data:
#             return jsonify({"error": "No JSON payload received"}), 400
        
#         return jsonify(output), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
