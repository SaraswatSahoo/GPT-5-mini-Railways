import os
import boto3
from botocore.exceptions import ClientError
import logging
import pickle
import tempfile
from typing import Optional, List

log = logging.getLogger(__name__)


class S3StorageManager:
    """Manages all S3 operations for PDFs, FAISS index, and images"""
    
    def __init__(self):
        """Initialize S3 client and bucket names from environment"""
        try:
            self.s3_client = boto3.client('s3')
            self.pdf_bucket = os.getenv('S3_PDF_BUCKET')
            self.knowledge_bucket = os.getenv('S3_KNOWLEDGE_BUCKET')
            
            if not self.pdf_bucket or not self.knowledge_bucket:
                raise ValueError("S3_PDF_BUCKET and S3_KNOWLEDGE_BUCKET must be set")
            
            log.info(f"S3 Manager initialized. PDF Bucket: {self.pdf_bucket}, Knowledge Bucket: {self.knowledge_bucket}")
        except Exception as e:
            log.error(f"Failed to initialize S3 manager: {e}")
            raise
    
    def download_pdf(self, pdf_key: str, local_path: str) -> str:
        """Download PDF from S3 to local temporary directory"""
        try:
            log.info(f"Downloading PDF from S3: {pdf_key}")
            self.s3_client.download_file(self.pdf_bucket, pdf_key, local_path)
            log.info(f"Successfully downloaded {pdf_key} to {local_path}")
            return local_path
        except ClientError as e:
            log.error(f"Failed to download PDF {pdf_key} from S3: {e}")
            raise
    
    def upload_image(self, local_path: str, s3_key: str) -> str:
        """Upload extracted image to S3"""
        try:
            log.info(f"Uploading image to S3: {s3_key}")
            self.s3_client.upload_file(local_path, self.knowledge_bucket, s3_key)
            s3_uri = f"s3://{self.knowledge_bucket}/{s3_key}"
            log.info(f"Image uploaded successfully: {s3_uri}")
            return s3_uri
        except ClientError as e:
            log.error(f"Failed to upload image {s3_key}: {e}")
            raise
    
    def upload_faiss_index(self, local_faiss_path: str, local_pkl_path: str) -> bool:
        """Upload FAISS index and mapping pickle file to S3"""
        try:
            log.info("Uploading FAISS index to S3...")
            
            self.s3_client.upload_file(
                local_faiss_path,
                self.knowledge_bucket,
                'faiss/index_hnsw.faiss'
            )
            log.info("FAISS index uploaded successfully")
            
            self.s3_client.upload_file(
                local_pkl_path,
                self.knowledge_bucket,
                'faiss/index_hnsw.faiss.pkl'
            )
            log.info("FAISS mapping uploaded successfully")
            
            return True
        except ClientError as e:
            log.error(f"Failed to upload FAISS index: {e}")
            raise
    
    def download_faiss_index(self, local_dir: str) -> bool:
        """Download FAISS index from S3 to local directory"""
        try:
            os.makedirs(local_dir, exist_ok=True)
            log.info("Downloading FAISS index from S3...")
            
            self.s3_client.download_file(
                self.knowledge_bucket,
                'faiss/index_hnsw.faiss',
                os.path.join(local_dir, 'index_hnsw.faiss')
            )
            
            self.s3_client.download_file(
                self.knowledge_bucket,
                'faiss/index_hnsw.faiss.pkl',
                os.path.join(local_dir, 'index_hnsw.faiss.pkl')
            )
            
            log.info("FAISS index downloaded successfully from S3")
            return True
        except ClientError as e:
            log.warning(f"FAISS index not found in S3: {e}")
            return False
    
    def download_image(self, s3_key: str, local_path: str) -> Optional[str]:
        """Download image from S3 to local path"""
        try:
            self.s3_client.download_file(self.knowledge_bucket, s3_key, local_path)
            return local_path
        except ClientError as e:
            log.error(f"Failed to download image {s3_key}: {e}")
            return None
    
    def list_pdfs(self) -> List[str]:
        """List all PDF files in S3 PDF bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.pdf_bucket)
            if 'Contents' not in response:
                log.info("No PDFs found in S3 bucket")
                return []
            
            pdfs = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith('.pdf')]
            log.info(f"Found {len(pdfs)} PDFs in S3 bucket")
            return pdfs
        except ClientError as e:
            log.error(f"Failed to list PDFs in S3: {e}")
            return []
    
    def get_pdf_download_url(self, pdf_key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for downloading a PDF"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.pdf_bucket, 'Key': pdf_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            log.error(f"Failed to generate presigned URL: {e}")
            raise


class LocalFileManager:
    """Fallback manager for local file operations (development mode)"""
    
    def __init__(self, pdf_dir: str = "./pdf", knowledge_dir: str = "./knowledge_pack"):
        """Initialize with local directories"""
        self.pdf_dir = pdf_dir
        self.knowledge_dir = knowledge_dir
        self.img_dir = os.path.join(knowledge_dir, "images")
        
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        
        log.info(f"Local file manager initialized. PDF dir: {self.pdf_dir}, Knowledge dir: {self.knowledge_dir}")
    
    def list_pdfs(self) -> List[str]:
        """List local PDF files"""
        pdfs = [
            os.path.join(self.pdf_dir, f) 
            for f in os.listdir(self.pdf_dir)
            if f.lower().endswith('.pdf')
        ]
        return pdfs


def get_storage_manager():
    """Factory function to get appropriate storage manager"""
    use_s3 = os.getenv('USE_S3', 'false').lower() == 'true'
    
    if use_s3:
        log.info("Using S3 storage")
        return S3StorageManager()
    else:
        log.info("Using local file storage")
        return LocalFileManager()