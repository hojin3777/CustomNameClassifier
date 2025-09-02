import { useState, useEffect, useCallback } from 'react';
import './App.css';

// Transaction 데이터의 타입을 정의합니다.
interface Transaction {
  id: number;
  trans_date: string;
  merchant: string;
  amount: number;
  category: string;
  balance: string;
}

function App() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [statusMessage, setStatusMessage] = useState('Loading transactions...');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const fetchTransactions = useCallback(async () => {
    try {
      setStatusMessage('Loading transactions...');
      const response = await fetch('http://localhost:5000/api/transactions');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data: Transaction[] = await response.json();
      setTransactions(data);
      if (data.length === 0) {
        setStatusMessage('No transactions found. Upload an image to get started!');
      }
    } catch (error) {
      console.error('Failed to fetch transactions:', error);
      setStatusMessage('Failed to load transactions. Is the backend server running?');
    }
  }, []);

  useEffect(() => {
    fetchTransactions();
  }, [fetchTransactions]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select an image file first.');
      return;
    }

    setIsUploading(true);
    setStatusMessage('Uploading and processing image...');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/api/ocr', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Image upload failed.');
      }
      
      const updatedTransactions: Transaction[] = await response.json();
      setTransactions(updatedTransactions);
      alert('Upload successful!');

    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred during upload.');
      setStatusMessage('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
      setSelectedFile(null);
    }
  };

  return (
    <div className="App">
      <h1>My Custom MyData</h1>
      
      <div className="upload-section">
        <h2>Upload New Transactions</h2>
        <input type="file" accept="image/*" onChange={handleFileChange} disabled={isUploading} />
        <button onClick={handleUpload} disabled={!selectedFile || isUploading}>
          {isUploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>

      <h2>Transaction List</h2>
      {transactions.length > 0 ? (
        // --- ✨ 여기가 테이블 전체 구조입니다 ---
        <table>
          {/* thead는 테이블의 제목 행입니다. */}
          <thead>
            <tr>
              <th>Date</th>
              <th>Merchant</th>
              <th>Category</th>
              <th>Amount</th>
            </tr>
          </thead>
          {/* tbody가 실제 데이터 목록이 들어가는 본문입니다. */}
          <tbody>
            {transactions.map((trans) => (
              <tr key={trans.id}>
                <td>{trans.trans_date || 'N/A'}</td>
                <td>{trans.merchant}</td>
                <td>{trans.category}</td>
                <td style={{ color: trans.amount < 0 ? 'blue' : 'red', textAlign: 'right' }}>
                  {(trans.amount || 0).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        // -----------------------------------------
      ) : (
        <p>{statusMessage}</p>
      )}
    </div>
  );
}

export default App;