#include "DNNResets.h"
#include "DNN.h"
#include <sys/time.h>
#include <gsl/gsl_blas.h>

using namespace flowstar;
using namespace std;

// this boolean is true after all DNNs have been loaded
bool dnn::dnn_initialized;

// DNN reachbility variables -- one per DNN
std::map<int, Continuous_Reachability_Setting> dnn::dnn_crs;
std::map<int, std::vector<iMatrix>> dnn::dnn_reset_weights;
std::map<int, std::vector<iMatrix>> dnn::dnn_reset_biases;
std::map<int, std::vector<dnn::activation>> dnn::dnn_activations;
std::map<int, std::vector<std::string>> dnn::augmentedVarNames;
std::vector<std::string> dnn::curAugmentedVarNames;
std::map<int, Variables> dnn::augmentedStateVars;
std::map<int, NNTaylorModelVec> dnn::activation_reset_map;

// DNN filenames
std::vector<std::string> dnn::DNN_Filenames;

// note that this reset only modifies the left Taylor model to make use of preconditioning
void linear_reset(NNTaylorModelVec & result, const iMatrix &reset_weights, const iMatrix &reset_biases,
		  const NNTaylorModelVec & tmv_left, const Continuous_Reachability_Setting & crs)
{

        //result = tmv_left;

	//std::vector<Interval> tmvPolyRange;
        //tmv_left.polyRange(tmvPolyRange, crs.step_end_exp_table);

	// don't need to truncate orders here since it's linear multiplication
	//bool ctrunc = false;
	//tmvReset.insert(result, tmv_left, tmvPolyRange, domain, crs.step_end_exp_table, crs.cutoff_threshold, ctrunc);

        tmv_left.linearTrans(result, reset_weights);
	result.addConstant(reset_biases);
	result.cutoff(crs.step_end_exp_table, crs.cutoff_threshold);
}

// note that this reset only modifies the left Taylor model to make use of preconditioning
void activation_reset(NNTaylorModelVec & result, const NNTaylorModelVec &tmvReset,
		      const NNTaylorModelVec & tmv_left, const Continuous_Reachability_Setting & crs)
{

        result = tmv_left;

	std::vector<Interval> tmvPolyRange;
        tmv_left.polyRange(tmvPolyRange, crs.step_end_exp_table);

	// need to truncate orders here since we're performing a polynomial reset
	bool ctrunc = true;
	bool cutoff = true;

	tmvReset.insert(result, tmv_left, tmvPolyRange, crs.step_end_exp_table,
			ctrunc, cutoff, crs.cutoff_threshold, crs.order);

}

void initialize_dnn(const Variables realStateVars){

	for (int dnnIndex = 0; dnnIndex < dnn::DNN_Filenames.size(); dnnIndex++){

	        std::string dnn_filename = dnn::DNN_Filenames[dnnIndex];

		std::vector<iMatrix> cur_dnn_reset_weights;
		std::vector<iMatrix> cur_dnn_reset_biases;
		std::vector<dnn::activation> cur_dnn_activations;
		std::vector<std::string> curAugmentedVarNames;
		std::vector<std::string> emptyVarNames; //used for legacy purposes in the load_dnn function
		Variables curAugmentedStateVars;
		
		dnn::load_dnn(cur_dnn_reset_weights, cur_dnn_reset_biases,
			      cur_dnn_activations, curAugmentedStateVars,
			      curAugmentedVarNames, realStateVars,
			      emptyVarNames, dnn_filename);


		Continuous_Reachability_Setting cur_dnn_crs;
		
		//Create a reachability setting for the DNN computations.
		cur_dnn_crs.setFixedStepsize(0.01);
		//NB: Verisig only perform a 3rd and 4th order reset atm (but using any order here is OK)
		cur_dnn_crs.setFixedOrder(4);
		cur_dnn_crs.setPrecision(100);
		Interval cutoff(-1e-14,1e-14);
		cur_dnn_crs.setCutoff(cutoff);
		
		Interval E(-0.1,0.1);
		std::vector<Interval> estimation;
		for(int i = 0; i < curAugmentedStateVars.size() - 1; ++i){
		        estimation.push_back(E);	// estimation for the i-th variable
		}
		cur_dnn_crs.setRemainderEstimation(estimation);
		
		cur_dnn_crs.prepareForReachability();
		//end of reachability setting initialization
		
		/*
		  Initialize the activation function reset TMV
		  This is originally the identity as it will be changing dynamically.
		*/
		NNTaylorModelVec tmv_activation_reset;
		for(int varInd = 0; varInd < curAugmentedVarNames.size() - 1; varInd++){
		  
		        NNTaylorModel tm_reset;
			if(!strncmp(curAugmentedVarNames[varInd+1].c_str(), "_f", strlen("_f"))){
		                tm_reset = NNTaylorModel("0", curAugmentedStateVars);
			}
			else{
		                tm_reset = NNTaylorModel(curAugmentedVarNames[varInd+1],
						       curAugmentedStateVars);
			}
			
			tmv_activation_reset.tms.push_back(tm_reset);
		}

		NNTaylorModelVec cur_activation_reset;
		cur_activation_reset = tmv_activation_reset;
		//end of activation reset initialization
		
		dnn::dnn_crs[dnnIndex + 1] = cur_dnn_crs;
		dnn::dnn_reset_weights[dnnIndex + 1] = cur_dnn_reset_weights;
		dnn::dnn_reset_biases[dnnIndex + 1] = cur_dnn_reset_biases;
		dnn::dnn_activations[dnnIndex + 1] = cur_dnn_activations;
		dnn::augmentedVarNames[dnnIndex + 1] = curAugmentedVarNames;
		dnn::augmentedStateVars[dnnIndex + 1] = curAugmentedStateVars;
		
		dnn::activation_reset_map[dnnIndex + 1] = cur_activation_reset;
		
	}

	//toggle the initialized bool
	dnn::dnn_initialized = true;
  
}

void print_tms(const TaylorModelVec &tmv, const std::vector<std::string> &curAugmentedVarNames)
{

        for(int varInd = 0; varInd < curAugmentedVarNames.size() - 1; varInd++){
	  
	        Polynomial poly = tmv.tms[varInd].expansion;
	        string printing;
		poly.toString(printing, curAugmentedVarNames);
						    
		printf("TM for %s: %s\n", curAugmentedVarNames[varInd+1].c_str(), printing.c_str());
		printf("remainder for %s: [%13.10f, %13.10f]\n", curAugmentedVarNames[varInd+1].c_str(),
		       tmv.tms[varInd].remainder.inf(),
		       tmv.tms[varInd].remainder.sup());
	}
}

void print_tms(const NNTaylorModelVec &tmv, const std::vector<std::string> &curAugmentedVarNames)
{

        for(int varInd = 0; varInd < curAugmentedVarNames.size() - 1; varInd++){
	        NNPolynomial poly = tmv.tms[varInd].expansion;
	        string printing;
		poly.toString(printing, curAugmentedVarNames);
						    
		printf("TM for %s: %s\n", curAugmentedVarNames[varInd+1].c_str(), printing.c_str());
		printf("remainder for %s: [%13.10f, %13.10f]\n", curAugmentedVarNames[varInd+1].c_str(),
		       tmv.tms[varInd].remainder.inf(),
		       tmv.tms[varInd].remainder.sup());
	}
}

void print_matrix(gsl_matrix * A){
        for(int i = 0; i < A->size1; i++){
	        printf("row %d: ", i+1);
		for(int j = 0; j < A->size2; j++){
		        printf("%f, ", gsl_matrix_get(A, i, j));
		}
		printf("\n");
	}
}

void identity_preconditioning(NNTaylorModelVec &tmv_left, NNTaylorModelVec &tmv_right, const NNTaylorModelVec &tmvImage,
			      const bool normalize, const Continuous_Reachability_Setting & crs)
{

	//NB: this assumes tmvImage is normalized within [-1,1], which
	//should be the case if this function is only used during DNN
	//resets

	// Compute the scaling matrix S.
	std::vector<Interval> S, invS;
	Interval intZero, intOne(1);

	std::vector<Interval> tmvPolyRange;
	tmvImage.polyRange(tmvPolyRange, crs.step_exp_table);

	for(int i=0; i<tmvPolyRange.size(); ++i)
	{
		Interval intSup;
		tmvPolyRange[i].mag(intSup);

		if(!normalize){
		        S.push_back(intOne);
			invS.push_back(intOne);
		}

		else if(intSup.subseteq(intZero))
		{
			S.push_back(intZero);
			invS.push_back(intOne);

		}
		else
		{
			S.push_back(intSup);
			Interval intRecSup;
			intSup.rec(intRecSup);
			invS.push_back(intRecSup);
		}
	}
	
	NNTaylorModelVec tmv_left_temp(S);

	NNTaylorModelVec tmv_right_temp = tmvImage;
        tmv_right_temp.scale_assign(invS);

	tmv_right = tmv_right_temp;
	tmv_left = tmv_left_temp;
}

//NB: the matrix R is not used anywhere, so we are keeping it in the function
void call_QR(gsl_matrix * A, gsl_matrix * Q)
{

	int m = A->size1;
	int n = A->size2;

	int k = std::min(m, n);

	// initialize some matrices used in the computation
	gsl_vector *tau = gsl_vector_calloc(k);
	gsl_matrix * Q_cur = gsl_matrix_calloc(m, m);
	gsl_matrix * Q_temp = gsl_matrix_calloc(m, m);
	gsl_matrix * eye = gsl_matrix_calloc(m, m);
	gsl_vector *v = gsl_vector_calloc(m);

	// call the QR function
  	gsl_linalg_QR_decomp(A, tau);

	// initialize Q at identity
	gsl_matrix_set_identity(Q);

	for(int j = 0; j < k; j++){

	        // create v
	        // elements above the diagonal are set to 0
	        for(int i = 0; i < j; i++){
		        gsl_vector_set(v, i, 0);
		}
		//diagonal is set to 1
		gsl_vector_set(v, j, 1);
		//elements below the diagonal are current values of A 
		for(int i = j+1; i < m; i++){
		        gsl_vector_set(v, i, gsl_matrix_get(A, i, j));
		}

		//Q_cur = tau_j * v_j * v_j'
		for(int ii = 0; ii < m; ii++){
		        for(int jj = 0; jj < m; jj++){
			        gsl_matrix_set(Q_cur, ii, jj, gsl_vector_get(tau, j) * gsl_vector_get(v, ii) * gsl_vector_get(v, jj));
			}
		}

		// eye = I
		gsl_matrix_set_identity(eye);

		// eye = I - Q_cur
		gsl_matrix_sub(eye, Q_cur);

		// Q_temp = eye * Q
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, eye, Q, 0.0, Q_temp);

		// Q = Q_temp
		gsl_matrix_memcpy(Q, Q_temp);
		
	}

	// free variables
	gsl_vector_free(tau);
	gsl_matrix_free(Q_cur);
	gsl_matrix_free(Q_temp);
	gsl_matrix_free(eye);
	gsl_vector_free(v);
}

//NB: num_plant_states is the number of states in the non-augmented hybrid system (not just plant states per se)
void qr_preconditioning(NNTaylorModelVec &tmv_left, NNTaylorModelVec &tmv_right, NNTaylorModelVec &tmv_composed,
			const NNTaylorModelVec &tmvImage, const int num_init_conditions,
			const vector<string> curAugmentedVarNames, const vector<Interval> domain,
			const Continuous_Reachability_Setting & crs)
{

        struct timeval begin;
	struct timeval end;

	gettimeofday(&begin, NULL);  

	// prepare new taylor models
	int num_total_states = tmvImage.tms.size();

        iMatrix constant_parts;
	tmvImage.constant(constant_parts);
	
	// construct the A matrix and the TMV of constants
	// NB: the matrix A only contains the first num_init_conditions variables because they are sufficient to construct Q
	// NB: the matrix A does not contain local_t -- it is added back later on as a column of 0s

	gsl_matrix * A = gsl_matrix_calloc(num_init_conditions, num_init_conditions);
	gsl_matrix_set_zero(A);

	// this loop collects constant parts AND populates A
	for (int i = 0; i < num_init_conditions; i++){

		for(auto iter=tmvImage.tms[i].expansion.monomials_map.begin(); iter != tmvImage.tms[i].expansion.monomials_map.end(); iter++){
			int index = 0;

			if ((iter->second)->isLinear(index)){
			        gsl_matrix_set(A, i, index-1, (iter->second)->getCoefficient().midpoint()); //the -1 removes time
			}
		}
	        
	}

	//NNTaylorModelVec constants(constant_parts, num_total_states + 1);

	// this is used when creating TMV from matrices that don't have the local_t column
	bool noTime = true;

	gsl_matrix * Q = gsl_matrix_calloc(num_init_conditions, num_init_conditions);
	call_QR(A, Q);

	gettimeofday(&end, NULL);
	double elapsedSecsQR = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR setup and QR call time: %f\n", elapsedSecsQR);			

	gettimeofday(&begin, NULL);
	// the Q_full matrix is a block-diagonal matrix [Q, 0;, 0, I]
	gsl_matrix * Q_full = gsl_matrix_calloc(num_total_states, num_total_states);
	gsl_matrix_set_identity(Q_full);

	for(int i = 0; i < num_init_conditions; i++){
	  
	        for(int j = 0; j < num_init_conditions; j++){
		        gsl_matrix_set(Q_full, i, j, gsl_matrix_get(Q, i, j));
		}
	}

	iMatrix MQ(Q_full);

	iMatrix iMQt;
	MQ.transpose(iMQt);

	iMatrix t;
	
	MQ.mul(t, iMQt);
	gettimeofday(&end, NULL);
	double elapsedSecsQRMat = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR matrix creation time: %f\n", elapsedSecsQRMat);

	gettimeofday(&begin, NULL);

	Real maxGap, maxGap2, minRowSum, bloatEps, zeroR;

	// first, get current gap
	t.getMaxGapIdentity(maxGap, t);
	gettimeofday(&end, NULL);
	double elapsedSecsQRGap = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR gap finding time: %f\n", elapsedSecsQRGap);

	gettimeofday(&begin, NULL);
	
	// prepare the right TMV first by subtracting the constant terms
	NNTaylorModelVec tmvInit = NNTaylorModelVec(tmvImage);
	tmvInit.rmConstant();
	//tmvImage.sub(tmvInit, constants);

	gettimeofday(&end, NULL);
	double elapsedSecsQRSub = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR tmv sub time: %f\n", elapsedSecsQRSub);

	gettimeofday(&begin, NULL);	
	// insert Q' matrix in right TMV
	std::vector<Interval> tmvPolyRange;
	tmvInit.polyRange(tmvPolyRange, crs.step_end_exp_table);
	gettimeofday(&end, NULL);
	double elapsedSecsQRPoly = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR poly range time: %f\n", elapsedSecsQRPoly);	


	gettimeofday(&begin, NULL);

	NNTaylorModelVec tmv;
     
	tmvInit.linearTrans(tmv, iMQt);
	tmv.cutoff(crs.step_end_exp_table, crs.cutoff_threshold);

	gettimeofday(&end, NULL);
	double elapsedSecsQRIns = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR insert Q time: %f\n", elapsedSecsQRIns);				

	gettimeofday(&begin, NULL);
	// Compute the scaling matrix S.
	std::vector<Interval> S, invS;
	Interval intZero, intOne(1);
	
	tmv.polyRange(tmvPolyRange, crs.step_end_exp_table);

	for(int i=0; i<tmvPolyRange.size(); ++i)
	{
		Interval intSup;
		tmvPolyRange[i].mag(intSup);

		if(intSup.subseteq(intZero))
		{
			S.push_back(intZero);
			invS.push_back(intOne);
		}
		else
		{
			S.push_back(intSup);
			Interval intRecSup;
			intSup.rec(intRecSup);
			invS.push_back(intRecSup);

		}
	}

	tmv.scale_assign(invS);
	
	// prepare the left TMV -- scale first


	gettimeofday(&begin, NULL);  
	//NNTaylorModelVec tmvPre(MQ, noTime);
	NNTaylorModelVec tmvPreTemp(S);
	NNTaylorModelVec tmvPre;
	tmvPreTemp.linearTrans(tmvPre, MQ);
	tmvPre.cutoff(crs.step_end_exp_table, crs.cutoff_threshold);
	
	gettimeofday(&end, NULL);
	double elapsedSecsJustIn = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR final insert (just TMV init) time: %f\n", elapsedSecsJustIn);		

	// add remainders to tmvPre
	Interval tempI;
	maxGap.to_sym_int(tempI);

	for(int i = 0; i < num_init_conditions; i++){
	        tmvPre.tms[i].remainder = Interval(tempI);
	}
	
	// add constant part
	//tmvPre.add_assign(constants);
	tmvPre.addConstant(constant_parts);

	gettimeofday(&end, NULL);
	double elapsedSecsQRFin = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR final insert time: %f\n", elapsedSecsQRFin);					

        tmv_right = tmv;
        tmv_left = tmvPre;

	gettimeofday(&begin, NULL);  
	// finally, compose again
	MQ.right_scale_assign(S);
	tmv_right.linearTrans(tmv_composed, MQ);
	//tmv_composed.add_assign(constant);
	tmv_composed.addConstant(constant_parts);
	for(int i = 0; i < num_init_conditions; i++){
	        tmv_composed.tms[i].remainder = Interval(tempI);
	}
	gettimeofday(&end, NULL);
	double elapsedSecsQRComp2 = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
	printf("elapsed QR second composition time: %f\n", elapsedSecsQRComp2);						
	
	//free gsl matrices
	gsl_matrix_free(A);
	gsl_matrix_free(Q);
	gsl_matrix_free(Q_full);
	
}



void dnn_reachability::compute_dnn_reachability(Flowpipe &result, const TaylorModelVec tmvImage,
						const Flowpipe fpAggregation, const std::string modeName,
						const std::vector<std::string> stateVarNames,
						const Variables realStateVars, const bool bPrint){

        if(!dnn::dnn_initialized){
	        if(bPrint) printf("Loading neural networks...\n");

		initialize_dnn(realStateVars);
	}

	result = fpAggregation;

	// first, figure out which network we're at
	int dnnIndex = 1;
					
	if (strlen(modeName.c_str()) > 3){

	        int startDnnIndex = 3;

		if(!strncmp(modeName.c_str(), "DNNm", strlen("DNNm"))){
		        startDnnIndex = 4;
		}
		
		dnnIndex = std::stoi(modeName.substr(startDnnIndex));
	}
					
	std::vector<dnn::activation> cur_dnn_activations = dnn::dnn_activations[dnnIndex];
	std::vector<iMatrix> cur_dnn_reset_weights = dnn::dnn_reset_weights[dnnIndex];
	std::vector<iMatrix> cur_dnn_reset_biases = dnn::dnn_reset_biases[dnnIndex];
	dnn::curAugmentedVarNames = dnn::augmentedVarNames[dnnIndex];
	Variables curAugmentedStateVars = dnn::augmentedStateVars[dnnIndex];
	Continuous_Reachability_Setting cur_dnn_crs = dnn::dnn_crs[dnnIndex];
	NNTaylorModelVec cur_activation_reset = dnn::activation_reset_map[dnnIndex];

	//all_ranges is used to store interval approximations of all states
	std::vector<Interval> all_ranges;

	int numDnnVars = dnn::curAugmentedVarNames.size();

	/*
	  I alternate between after_activation_reset and after_linear_reset
	  for consistency with the initial autogenerated C++ code
	*/
        NNTaylorModelVec tmv_left_after_activation_reset;
        NNTaylorModelVec tmv_left_after_linear_reset;
	NNTaylorModelVec tmv_right;

	std::vector<Interval> domain;

	// Create flowpipes for all new states
	//after_activation_reset.domain.push_back(fpAggregation.domain[0]);
	domain.push_back(fpAggregation.domain[0]);

	// add _f states from the plant reachability problem first
	std::map<int, NNTaylorModel> tmvMap;
	std::map<int, NNTaylorModel> tmvPreMap;
	std::map<int, Interval> domainMap;

	int numFstates = 0;

	for(int varInd = 0; varInd < stateVarNames.size(); varInd++){

	        if(!strncmp(stateVarNames[varInd].c_str(), "_f", strlen("_f"))){

		        int fIndex = std::stoi(stateVarNames[varInd].substr(2));

			tmvMap[fIndex] = NNTaylorModel(fpAggregation.tmv.tms[varInd]);
			tmvPreMap[fIndex] = NNTaylorModel(fpAggregation.tmvPre.tms[varInd]);
			domainMap[fIndex] = fpAggregation.domain[varInd+1];

			numFstates++;
		}
	}

	std::map<int, int> stateToF, fToState;
	dnn::get_state_to_f_map(stateToF, fToState, stateVarNames, tmvImage);
					
	for(int varInd = 0; varInd < numFstates; varInd++){

	        NNTaylorModel tmPre, tm;
	  
		dnn::convert_TM_dimension(tmPre, tmvPreMap[varInd+1], numDnnVars, varInd, stateToF);
		dnn::convert_TM_dimension(tm, tmvMap[varInd+1], numDnnVars, varInd);  // should be identity

		tmv_left_after_activation_reset.tms.push_back(NNTaylorModel(tmPre));
		tmv_right.tms.push_back(NNTaylorModel(tm));
		domain.push_back(domainMap[varInd+1]);
					  
	}
	
	Interval intZero(0.0, 0.0);
	for(int varInd = numFstates; varInd < dnn::curAugmentedVarNames.size() - 1; varInd++){

	        tmv_left_after_activation_reset.tms.push_back(NNTaylorModel(intZero, dnn::curAugmentedVarNames.size()));
	        tmv_right.tms.push_back(NNTaylorModel(intZero, dnn::curAugmentedVarNames.size()));
		domain.push_back(Interval(-1.0, 1.0));
	}

	// precondition all TMs just in case
	NNTaylorModelVec tmv_composed;
	std::vector<Interval> tmvPolyRange;
	tmv_right.polyRange(tmvPolyRange, cur_dnn_crs.step_end_exp_table);	
	tmv_left_after_activation_reset.insert(tmv_composed, tmv_right, tmvPolyRange,
					       cur_dnn_crs.step_end_exp_table,
					       true, true, cur_dnn_crs.cutoff_threshold, cur_dnn_crs.order);

	bool normalize = true;
	identity_preconditioning(tmv_left_after_activation_reset, tmv_right, tmv_composed, normalize, cur_dnn_crs);


	//go through the DNN resets
	for(int layer = 0; layer < cur_dnn_reset_weights.size(); layer++){


	        if(bPrint){
	                printf("Jumping to layer %d\n", layer + 1);
			printf("Performing linear reset...\n");
		}

		struct timeval begin;
		struct timeval end;

		gettimeofday(&begin, NULL);
		
		//NB: this assumes there is always a linear reset first
		linear_reset(tmv_left_after_linear_reset, cur_dnn_reset_weights[layer], cur_dnn_reset_biases[layer],
			     tmv_left_after_activation_reset, cur_dnn_crs);

		gettimeofday(&end, NULL);
		double elapsedSecsLIN = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
		printf("elapsed linear reset time: %f\n", elapsedSecsLIN);

		//if no activation function, just set after_activation_reset equal to after_linear_reset
		if(cur_dnn_activations[layer] != dnn::SIGMOID &&
		   cur_dnn_activations[layer] != dnn::TANH &&
		   cur_dnn_activations[layer] != dnn::SWISH &&
		   cur_dnn_activations[layer] != dnn::RELU){
		  
	                tmv_left_after_activation_reset = tmv_left_after_linear_reset;
			continue;
		}


	        if(bPrint){
			printf("Performing preconditioning...\n");
		}

		gettimeofday(&begin, NULL);	
		
		// first compose
		NNTaylorModelVec tmv_composed;
		std::vector<Interval> tmvPolyRange;
		tmv_right.polyRange(tmvPolyRange, cur_dnn_crs.step_end_exp_table);
		
		tmv_left_after_linear_reset.insert(tmv_composed, tmv_right, tmvPolyRange,
						   cur_dnn_crs.step_end_exp_table,
						   true, true, cur_dnn_crs.cutoff_threshold, cur_dnn_crs.order);

		// print_tms(tmv_composed, curAugmentedVarNames);
		// if(layer == 1)
		//  exit(1);		

		gettimeofday(&end, NULL);
		double elapsedSecsQR = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
		printf("elapsed QR composition time: %f\n", elapsedSecsQR);		

		// this also composes the new left and right again since preconditioning might add numeric error
		qr_preconditioning(tmv_left_after_linear_reset, tmv_right, tmv_composed, tmv_composed,
				   numFstates, dnn::curAugmentedVarNames, domain, cur_dnn_crs);
				
		tmv_composed.intEvalNormal(all_ranges, cur_dnn_crs.step_end_exp_table);
				
		bool tanh_act;					
		if(cur_dnn_activations[layer] == dnn::SIGMOID){
	                tanh_act = false;
		}						
		else if(cur_dnn_activations[layer] == dnn::TANH){
	                tanh_act = true;
		}						
		else{
	                printf("Verisig only supports tanh and sigmoid currently.\n");
			exit(1);
		}

		//modify the reset depending on the activation function TM
		for(int varInd = 0; varInd < dnn::curAugmentedVarNames.size() - 1; varInd++){

		        if(strncmp(dnn::curAugmentedVarNames[varInd+1].c_str(), "_f", strlen("_f"))) continue;

			Interval intC = all_ranges[varInd];

			NNTaylorModel new_tm_reset;
 
			dnn::act_reset(new_tm_reset, intC, varInd, numDnnVars, tanh_act, 4);
			
			cur_activation_reset.tms[varInd] = new_tm_reset;

		}
						
		if(bPrint){
	               printf("Performing activation reset...\n");
		}

		gettimeofday(&begin, NULL);		
		//perform the activation reset
		activation_reset(tmv_left_after_activation_reset, cur_activation_reset,
				 tmv_left_after_linear_reset, cur_dnn_crs);

		// print_tms(tmv_left_after_activation_reset, dnn::curAugmentedVarNames);
		// printf("right\n");
		// print_tms(tmv_right, dnn::curAugmentedVarNames);
		// if(layer == 1)
		//  exit(1);

		gettimeofday(&end, NULL);
		double elapsedSecsACT = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
		printf("elapsed activation reset time: %f\n", elapsedSecsACT);
	}


	// precondition again just in case
	tmv_right.polyRange(tmvPolyRange, cur_dnn_crs.step_end_exp_table);
	tmv_left_after_activation_reset.insert(tmv_composed, tmv_right, tmvPolyRange,
					       cur_dnn_crs.step_end_exp_table,
					       true, true, cur_dnn_crs.cutoff_threshold, cur_dnn_crs.order);
        normalize = false;
	identity_preconditioning(tmv_left_after_activation_reset, tmv_right, tmv_composed, normalize, cur_dnn_crs);
					
	// Store DNN flowpipes back in the plant flowpipe
	for(int varInd = 0; varInd < stateVarNames.size(); varInd++){
	  
	        if(!strncmp(stateVarNames[varInd].c_str(), "_f", strlen("_f"))){

		        NNTaylorModel tmPre, tm;

			int fIndex = std::stoi(stateVarNames[varInd].substr(2));

			//NB: left and right are switched since the Flow* representation is flipped for some reason
			dnn::convert_TM_dimension(tm, tmv_left_after_activation_reset.tms[fIndex - 1],
						  stateVarNames.size() + 1, varInd); //should be identity
			dnn::convert_TM_dimension(tmPre, tmv_right.tms[fIndex - 1],
						  stateVarNames.size() + 1, varInd, fToState);
			
			result.tmvPre.tms[varInd] = TaylorModel(tmPre);
			result.tmv.tms[varInd] = TaylorModel(tm);
			result.domain[varInd+1] = domain[varInd+1];
		}
	}

	// printing
	//add time
	// std::vector<std::string> realVarNames;
	// realVarNames.push_back("local_t");

	// for(int i = 0; i < stateVarNames.size(); i++){
	//   realVarNames.push_back(stateVarNames[i]);
	// }

	// print_tms(result.tmvPre, realVarNames);
	// printf("right\n");
	// print_tms(result.tmv, realVarNames);
	//exit(1);
        
}
