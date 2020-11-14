package PseudoRFSearch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Classes.Document;
import Classes.Query;
import IndexingLucene.MyIndexReader;
import SearchLucene.*;
//import Search.*;

public class PseudoRFRetrievalModel {
	MyIndexReader ixreader;
	private final long totalClen;
    private final QueryRetrievalModel3 model3;
    // set mu = 2000, Dirichlet Prior Smoothing, most case mu = 2000
    private final double mu = 2000;

	public PseudoRFRetrievalModel(MyIndexReader ixreader) {
		this.ixreader=ixreader;
		this.totalClen = ixreader.getTotalContentLength();
        this.model3 = new QueryRetrievalModel3(ixreader);
	}

	/**
	 * Search for the topic with pseudo relevance feedback in 2020 Fall assignment 4.
	 * The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
	 *
	 * @param aQuery The query to be searched for.
	 * @param TopN The maximum number of returned document
	 * @param TopK The count of feedback documents
	 * @param alpha parameter of relevance feedback model
	 * @return TopN most relevant document, in List structure
	 */
	public List<Document> RetrieveQuery( Query aQuery, int TopN, int TopK, double alpha) throws Exception {	
		// this method will return the retrieval result of the given Query, and this result is enhanced with pseudo relevance feedback
		// (1) you should first use the original retrieval model to get TopK documents, which will be regarded as feedback documents
		// (2) implement GetTokenRFScore to get each query token's P(token|feedback model) in feedback documents
		// (3) implement the relevance feedback model for each token: combine the each query token's original retrieval score P(token|document) with its score in feedback documents P(token|feedback model)
		// (4) for each document, use the query likelihood language model to get the whole query's new score, P(Q|document)=P(token_1|document')*P(token_2|document')*...*P(token_n|document')


		//get P(token|feedback documents)
		HashMap<String,Double> TokenRFScore = GetTokenRFScore(aQuery,TopK);

		// sort all retrieved documents from most relevant to least, and return TopN
		List<Document> results = new ArrayList<Document>();

        // get postings, model3 is last Assignment model 
        HashMap<Integer, HashMap<String, Integer>> postings = this.model3.getQueryResult();
        String[] tokens = aQuery.GetQueryContent().split(" ");

        // Relevance Feedback Model
        for (int docid : postings.keySet()) {
            HashMap<String, Integer> postings_store = postings.get(docid);
            long docLen = this.ixreader.docLength(docid);
            double score = 1.0;
            double len = (docLen + mu);
            // |D|/(|D| + mu), mu/(|D| + mu)
            double coeff1 = 1.0 * docLen / len, coeff2 = 1.0 * mu / len;
            for (String token : tokens) {
            	long cf = this.ixreader.CollectionFreq(token);
            	if (cf == 0) continue;  
            	long tf = postings_store.getOrDefault(token, 0);
            	//c(w, D)/|D|, p(w, REF)
            	double left = 1.0 * tf / docLen, right = 1.0 * cf / this.totalClen;
                // Unigram Language Model
                score *= alpha * (coeff1 * left + coeff2 * right) + (1 - alpha) * TokenRFScore.getOrDefault(token, 0.0);
            }
            results.add(new Document(String.valueOf(docid), ixreader.getDocno(docid), score));
        }

        results.sort((d1, d2) -> d1.score() > d2.score() ? -1 : 1);
        return results.subList(0, TopN);
	}

	public HashMap<String,Double> GetTokenRFScore(Query aQuery,  int TopK) throws Exception
	{
		// for each token in the query, you should calculate token's score in feedback documents: P(token|feedback documents)
		// use Dirichlet smoothing
		// save <token, score> in HashMap TokenRFScore, and return it
		HashMap<String,Double> TokenRFScore=new HashMap<String,Double>();
		 List<Document> originalRes = this.model3.retrieveQuery(aQuery, TopK);
	        HashMap<Integer, HashMap<String, Integer>> p = this.model3.getQueryResult();
	        HashMap<String, Long> pDoc = new HashMap<>();
	        
	        long docLen = populatePseudoDoc(pDoc, originalRes, p);
            double len = (docLen + mu);
            // |D|/(|D| + mu), mu/(|D| + mu)
            double coeff1 = 1.0 * docLen / len, coeff2 = 1.0 * mu / len;
            for (String token : pDoc.keySet()) {
            	long cf = this.model3.getCollectionFreq(token);
            	if (cf == 0) continue;  
            	long tf = pDoc.getOrDefault(token, 0L);
            	//c(w, D)/|D|, p(w, REF)
            	double left = 1.0 * tf / docLen, right = 1.0 * cf / this.totalClen;
	            // Unigram Language Model
	            TokenRFScore.put(token, coeff1 * left + coeff2 * right);
	        }
		return TokenRFScore;
	}
	
	private long populatePseudoDoc(HashMap<String, Long> doc, List<Document> orig, HashMap<Integer, HashMap<String, Integer>> postings) throws IOException {
        long totalLen = 0;
        for (Document d : orig) {
            int docId = Integer.valueOf(d.docid());
            HashMap<String, Integer> posting = postings.get(docId);
            if (posting == null) {
                System.err.println("Document not exists");
                continue;
            }
            for (String token : posting.keySet()) {
                int tf = posting.getOrDefault(token, 0);
                long old = doc.getOrDefault(token, 0L);
                doc.put(token, old + tf);
            }
            totalLen += this.ixreader.docLength(docId);
        }
        return totalLen;
    }


}
