(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b4 b0)
		(in-tower b4)
		(on b5 b4)
		(in-tower b5)
		(on b9 b5)
		(in-tower b9)
		(blue b4)
		(red b5)
		(maroon b5)
		(red b6)
		(green b6)
		(maroon b6)
		(blue b9)
		(blue b0)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b6 b5)) (not (on b8 b9)))))
)