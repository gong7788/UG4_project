(define (problem blocks-problem)
	(:domain blocksworld-unstack)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0 tower0)
	(:init 
		(arm-empty )
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(clear b7)
		(done t0)
		(in-tower t0 tower0)
		(done tower0)
		(tower tower0)
		(on b9 t0)
		(in-tower b9 tower0)
		(done b9)
		(on b8 b9)
		(in-tower b8 tower0)
		(done b8)
		(on-table b7)
		(on b2 b8)
		(in-tower b2 tower0)
		(done b2)
		(on b0 b2)
		(in-tower b0 tower0)
		(done b0)
		(on b1 b0)
		(in-tower b1 tower0)
		(done b1)
		(on-table b3)
		(clear b1)
		(purple b3)
		(pink b4)
		(= (red-count tower0) 0)
		(= (blue-count tower0) 0)
		(= (green-count tower0) 0)
		(= (yellow-count tower0) 0)
		(= (purple-count tower0) 0)
		(= (orange-count tower0) 0)
		(= (pink-count tower0) 0)
	)
	(:goal (and (forall (?x) (done ?x)) (forall (?x) (or (not (purple ?x)) (exists (?y) (and (pink ?y) (on ?x ?y))))) (and (not (on b7 b8)) (not (on b3 b1)))))
)