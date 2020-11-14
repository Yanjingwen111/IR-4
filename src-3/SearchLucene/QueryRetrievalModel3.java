package SearchLucene;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import Classes.Query;
import Classes.Document;
import IndexingLucene.MyIndexReader;

public class QueryRetrievalModel3 {
	
	protected MyIndexReader indexReader;;
    private final long collectionLen;
    private HashMap<String, Long> frequency = new HashMap<>();
    private HashMap<String, int[][]> postings = new HashMap<>();
    // set mu = 2000, Dirichlet Prior Smoothing, most case mu = 2000
    private double mu = 2000;
    private HashMap<Integer, HashMap<String, Integer>> queryResult;
	
	public QueryRetrievalModel3(MyIndexReader ixreader) {
		indexReader = ixreader;
        this.collectionLen = indexReader.getTotalContentLength();
	}
	
	/**
	 * Search for the topic information. 
	 * The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
	 * TopN specifies the maximum number of results to be returned.
	 * 
	 * @param aQuery The query to be searched for.
	 * @param TopN The maximum number of returned document
	 * @return
	 */
	
	/**
	 * From Assignment 3
	 */
	
	public List<Document> retrieveQuery( Query aQuery, int TopN ) throws IOException {
		// NT: you will find our IndexingLucene.Myindexreader provides method: docLength()
		// implement your retrieval model here, and for each input query, return the topN retrieved documents
		// sort the docs based on their relevance score, from high to low
		String[] queryTk = aQuery.GetQueryContent().split(" ");
        if (queryTk.length == 0) {
        	return new ArrayList<>(0);
        }
        return rankResult(queryTk, TopN);
	}
	 public HashMap<Integer, HashMap<String, Integer>> getQueryResult() {
	        return this.queryResult;
	    }
	//rank result based on scores
	private List<Document> rankResult (String[] tokens, int topN) throws IOException {
        HashMap<Integer, HashMap<String, Integer>> qResult = popqResult(tokens);
        ArrayList<Document> Result = queryLikelihood(qResult, tokens);
        this.queryResult = popqResult(tokens);
        //comparator: descend
        Result.sort((doc1, doc2) -> {
            double s1 = doc1.score(), s2 = doc2.score();
            return s1 > s2 ? -1 : 1;
        });

        int size = Math.min(topN, Result.size());
        ArrayList<Document> res = new ArrayList<>(size);
        int docCount = 0;
        for (Document re : Result) {
            res.add(re);
            if (++docCount > size - 1) {
            	break;
            } 
        }
        qResult.forEach((id, temp) -> temp.clear());
        qResult.clear();
        Result.clear();
        return res;
    }

    private HashMap<Integer, HashMap<String, Integer>> popqResult(String[] tokens) throws IOException {
        HashMap<Integer, HashMap<String, Integer>> CollectionTK = new HashMap<>();
        for (String token : tokens) {
            Long cf = getCollectionFreq(token);
            if (cf.equals(0L)) continue;
            int[][] postList = getCollectionPostings(token);
            for (int[] postDoc : postList) {
                int docid = postDoc[0], docFreq = postDoc[1];
                HashMap<String, Integer> oneTF = CollectionTK.getOrDefault(docid, new HashMap<>());
                if (oneTF.size() == 0) {
                	CollectionTK.putIfAbsent(docid, oneTF);
                }
                oneTF.put(token, docFreq);
            }
        }
        return CollectionTK;
    }

    private int[][] getCollectionPostings(String token) throws IOException {
        if (!this.postings.containsKey(token)) {
            int[][] postingList = this.indexReader.getPostingList(token);
            if (postingList == null) {
            	postingList = new int[0][];
            } 
            if (postingList.length == 0) {
            	System.err.println(String.format("warning: <%s> is not in collection", token));
            }       
            this.postings.put(token, postingList);
        }
        return this.postings.get(token);
    }
    
    //calculate collection frequency
    private Long calCF(int[][] postings) {
        long count = 0;
        for (int[] po : postings) {
            count += po[1];
        }
        return count;
    }
    
    // calculate term frequency of given collection of given token
    public Long getCollectionFreq(String token) throws IOException {
        if (!this.frequency.containsKey(token)) {
            Long tf = this.indexReader.CollectionFreq(token);
            Long myFreq = calCF(getCollectionPostings(token));
            if (!myFreq.equals(tf)) {
            	System.err.println("CF not equal");
            }      
            this.frequency.put(token, tf);
        }
        return this.frequency.get(token);
    }

    private ArrayList<Document> queryLikelihood(HashMap<Integer, HashMap<String, Integer>> queryResult, String[] tokens)
            throws IOException {
        ArrayList<Document> allResults = new ArrayList<>(queryResult.size());
        for (Integer docid : queryResult.keySet()) {
            double score = calScore(tokens, queryResult.get(docid), this.indexReader.docLength(docid));
            Document d = new Document(docid.toString(), this.indexReader.getDocno(docid), score);
            allResults.add(d);
        }
        return allResults;
    }
    
    //calculate score
    private double calScore(String[] tokens, HashMap<String, Integer> doctfList, int docLength) {
        double score = 1.0;
        double len = (docLength + mu);
        // |D|/(|D| + mu), mu/(|D| + mu)
        double coeff1 = 1.0 * docLength / len, coeff2 = 1.0 * mu / len;
        for (String token : tokens) {
            Long cf = frequency.get(token);
            if (cf.equals(0L)) {
            	continue;
            }
            int tf = doctfList.getOrDefault(token, 0);
            //c(w, D)/|D|, p(w, REF)
            double left = 1.0 * tf / docLength, right = 1.0 * cf / collectionLen;
            //score = (|D|/(|D| + mu)*c(w, D)/|D|) + (mu/(|D| + mu)*p(w, REF))
            score *= (coeff1 * left + coeff2 * right);
        }
        //check whether score > 0
        if (score <= 0) {
        	score = 0;
        }
        return score;
    }
	
}