(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b5 b1)
		(in-tower b5)
		(on b2 b5)
		(in-tower b2)
		(on b3 b2)
		(in-tower b3)
		(on b4 b3)
		(in-tower b4)
		(on b8 b4)
		(in-tower b8)
		(on b9 b8)
		(in-tower b9)
		(blue b1)
		(blue b3)
		(red b5)
		(red b6)
		(red b7)
		(blue b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b6 b3)) (not (on b6 b9)))))
)