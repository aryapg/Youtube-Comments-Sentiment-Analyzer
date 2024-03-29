import React, { useState } from 'react';
import backgroundImage from './imageee.jpg';
import uploadIcon from './upload_icon.png';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileUploadMessage, setFileUploadMessage] = useState('');
  const [complianceResult, setComplianceResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [summaryText, setSummaryText] = useState('');
  const [topFeatures, setTopFeatures] = useState([]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setFileUploadMessage('');
    setComplianceResult('');
  };

  const checkCompliance = async () => {
    if (!selectedFile) {
      setFileUploadMessage("⚠️ Please upload a file before checking compliance.");
      return;
    }
  
    const formData = new FormData();
    formData.append('file', selectedFile);
  
    try {
      setIsLoading(true);
  
      const response = await fetch('http://localhost:5000/', {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error:', errorData.error);
        setFileUploadMessage(`Error: ${errorData.error}`);
        return;
      }
  
      const jsonData = await response.json();
      console.log('JSON Data:', jsonData);
  
      if (jsonData.complianceStatus) {
        setComplianceResult(`Compliance Status: ${jsonData.complianceStatus}`);
      }
      if (jsonData.topFeatures && jsonData.summaryText) {
        setTopFeatures(jsonData.topFeatures);
  
        // Process the summary text directly in the return statement
        const lines = jsonData.summaryText.split('\n').filter(line => line.trim() !== '');
        setSummaryText(
          <div style={{ textAlign: 'left' }}>
            {lines.map((line, index) => (
              <p key={index}>{line.includes(':') ? <span><strong>{line.split(':')[0]}</strong>: {line.split(':')[1]}</span> : line}</p>
            ))}
          </div>
        );
      }
  
      setFileUploadMessage('File uploaded successfully!');
    } catch (error) {
      console.error('Error:', error);
      setFileUploadMessage('An error occurred while checking compliance.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container" style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      backgroundImage: `url(${backgroundImage})`,
      backgroundSize: 'cover',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div className="content-container" style={{
        width: '600px',
        backgroundColor: 'rgba(255, 255, 255, 0.5)',
        padding: '40px',
        borderRadius: '20px',
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
        transition: 'transform 0.3s, box-shadow 0.3s, background-color 0.3s',
        backdropFilter: 'blur(10px)',
        textAlign: 'center',
      }}>
        <h1 style={{
          color: '#00008b',
          textAlign: 'center',
          marginBottom: '20px',
          fontSize: '32px',
          fontWeight: 'bold'
        }}>🛡️ Company Compliance Checker 🕵️</h1>
        <p style={{
          color: '#000000',
          textAlign: 'center',
          marginBottom: '30px',
          fontSize: '18px'
        }}>Hello! Please upload a dataset to check company compliance.</p>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          marginBottom: '20px',
          fontFamily: 'Tahoma, sans-serif',
        }}>
          <label htmlFor="fileInput" style={{
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            backgroundColor: '#007bff',
            color: '#fff',
            padding: '18px 25px',
            borderRadius: '30px',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
            transition: 'background-color 0.3s, transform 0.3s',
            marginBottom: '15px'
          }}>
            <img src={uploadIcon} alt="Upload Icon" style={{ marginRight: '10px', height: '20px', width: '20px' }} />
            Upload Dataset
          </label>
          <input
            type="file"
            id="fileInput"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          {selectedFile && (
            <p style={{
              color: '#555',
              marginTop: '10px',
              textAlign: 'center',
              fontSize: '16px'
            }}>Selected File: {selectedFile.name}</p>
          )}
          <button onClick={checkCompliance}
            style={{
              padding: '15px 70px',
              backgroundColor: '#000',
              color: '#fff',
              border: 'none',
              borderRadius: '30px',
              cursor: 'pointer',
              transition: 'background-color 0.3s, color 0.3s, transform 0.3s',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
              fontSize: '18px',
              fontWeight: 'bold',
              marginTop: '15px'
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = '#fff';
              e.target.style.color = '#000';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = '#000';
              e.target.style.color = '#fff';
            }}
          >
            Check Compliance
          </button>
        </div>
        {complianceResult && (
          <div style={{
            marginTop: '30px',
            padding: '16px',
            borderRadius: '10px',
            backgroundColor: complianceResult.includes('Not Compliant') ? '#d11b2c' : '#cff7cd',
            border: '1px solid #f5c6cb',
            transition: 'background-color 0.3s, border-color 0.3s'
          }}>
            <p style={{
              color: complianceResult.includes('Not Compliant') ? '#eef2ed' : '#155724',
              fontWeight: 'bold',
              fontSize: complianceResult.includes('Not Compliant') ? '24px' : '20px',
              marginBottom: '8px',
              textAlign: 'center',
              lineHeight: '1.4'
            }}>{complianceResult}</p>
          </div>
        )}
        {summaryText && (
          <div style={{ marginTop: '30px'}}>
            <h3>Summary Text:</h3>
            {summaryText}
          </div>
        )}
        {topFeatures.length > 0 && (
          <div style={{ marginTop: '30px', textAlign: 'left' }}>
            <h3 style={{ color: '#000', marginBottom: '10px' }}>Top Features:</h3>
            <ul style={{ listStyleType: 'disc', paddingLeft: '20px', fontSize: '16px' }}>
              {topFeatures.map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </ul>
          </div>
        )}
        {fileUploadMessage && (
          <p style={{
            color: fileUploadMessage.includes('successfully') ? '#008000' : '#ff0000',
            marginTop: '20px',
            textAlign: 'center',
            fontSize: fileUploadMessage.includes('successfully') ? '20px' : '18px',
            fontWeight: fileUploadMessage.includes('successfully') ? 'normal' : 'bold'
          }}>{fileUploadMessage}</p>
        )}
      </div>
    </div>
  );
};

export default App;
